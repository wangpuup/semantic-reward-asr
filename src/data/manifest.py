"""Build jsonl manifests from ESPnet/Kaldi dump dirs (wav.scp, text, utt2spk, utt2num_samples)."""
import argparse, json, os
import yaml
from src.data.normalize import normalize


def read_kv(path):
    out = {}
    with open(path) as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition(" ")
            out[k] = v
    return out


def build(cfg, split, spec):
    root = cfg["data"]["kaldi_root"]
    d = os.path.join(root, spec["kaldi_dir"])
    wav, text, spk = read_kv(f"{d}/wav.scp"), read_kv(f"{d}/text"), read_kv(f"{d}/utt2spk")
    nsamp = read_kv(f"{d}/utt2num_samples")
    max_samples = int(cfg["data"]["max_seconds"] * 16000)
    rows = []
    for utt in sorted(wav):
        if spec.get("exclude_prefix") and utt.startswith(spec["exclude_prefix"]):
            continue
        if int(nsamp[utt]) > max_samples:
            continue
        rows.append({"utt_id": utt, "wav": wav[utt], "n_samples": int(nsamp[utt]),
                     "text": normalize(text[utt]), "speaker": spk[utt], "split": split,
                     "feat": os.path.join(cfg["data"]["feature_dir"], split, utt + ".npy")})
        if len(rows) >= spec["max_utts"]:
            break
    os.makedirs(cfg["data"]["manifest_dir"], exist_ok=True)
    out = os.path.join(cfg["data"]["manifest_dir"], f"{split}.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{split}: {len(rows)} utts -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for split, spec in cfg["data"]["splits"].items():
        build(cfg, split, spec)
