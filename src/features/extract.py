"""Run the frozen speech encoder once and cache downsampled fp16 features per utterance.

Whisper encoder gives 1500 frames @ 50 Hz for a 30 s padded window. We keep only the frames that
cover real audio, stack `stack` consecutive frames (-> 10 Hz, dim 1280*stack) and save fp16 .npy.
"""
import argparse, json, os, sys
import numpy as np, torch, yaml, kaldiio
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel


def load_wav(kaldi_root, wav_spec):
    # wav.scp entries are "path/to/data_wav.ark:offset" relative to the recipe dir
    path, _, off = wav_spec.partition(":")
    sr, wav = kaldiio.load_mat(f"{os.path.join(kaldi_root, path)}:{off}")
    assert sr == 16000, sr
    return wav.astype(np.float32) / 32768.0 if wav.dtype == np.int16 else wav.astype(np.float32)


def stack_frames(x, k):
    T = (x.shape[0] // k) * k
    x = x[:T]
    return x.reshape(T // k, -1)


@torch.no_grad()
def main(cfg, splits):
    dev = "cuda"
    name, k, bs = cfg["encoder"]["name"], cfg["encoder"]["stack"], cfg["encoder"]["batch_size"]
    fe = WhisperFeatureExtractor.from_pretrained(name)
    enc = WhisperModel.from_pretrained(name, torch_dtype=torch.float16).encoder.to(dev).eval()
    for split in splits:
        rows = [json.loads(l) for l in open(f"{cfg['data']['manifest_dir']}/{split}.jsonl")]
        os.makedirs(os.path.join(cfg["data"]["feature_dir"], split), exist_ok=True)
        rows = [r for r in rows if not os.path.exists(r["feat"])]
        print(f"{split}: extracting {len(rows)} utts", flush=True)
        for i in tqdm(range(0, len(rows), bs), file=sys.stdout):
            batch = rows[i:i + bs]
            wavs = [load_wav(cfg["data"]["kaldi_root"], r["wav"]) for r in batch]
            feats = fe(wavs, sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float16)
            h = enc(feats).last_hidden_state  # (B, 1500, 1280)
            for r, w, hh in zip(batch, wavs, h):
                n_frames = int(np.ceil(len(w) / 16000 * 50))
                np.save(r["feat"], stack_frames(hh[:n_frames].float().cpu().numpy(), k).astype(np.float16))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"]); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    main(cfg, a.splits)
