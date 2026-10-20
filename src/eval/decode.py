"""Greedy decoding of a split with a saved checkpoint -> hypotheses jsonl + metrics json."""
import argparse, json, os, sys, time
import numpy as np, torch, yaml
from tqdm import tqdm
from sentence_transformers import SentenceTransformer   # imported at start so a broken env fails before training, not at first eval
from src.data.normalize import normalize
from src.model.speech_llm import SpeechLLM, load_manifest, load_feats, feat_path
from src.eval.metrics import evaluate


@torch.no_grad()
def decode_split(model, rows, bs, max_new_tokens):
    model.eval()
    hyps = []
    for i in tqdm(range(0, len(rows), bs), file=sys.stdout, desc="decode"):
        feats = load_feats(rows[i:i + bs])
        out = model.generate(feats, max_new_tokens=max_new_tokens)
        hyps += [normalize(model.decode(o)) for o in out]
    return hyps


_JUDGE = {}


def get_judge(cfg):
    """Evaluation-only LLM judge (cfg.eval.judge_model); cached across evals. None if not configured."""
    name = cfg["eval"].get("judge_model")
    if not name:
        return None
    if name not in _JUDGE:
        from src.rewards.registry import LLMJudgeReward
        _JUDGE[name] = LLMJudgeReward(model=name)
    return _JUDGE[name]


def run_eval(model, cfg, split, out_dir, tag="", bertscore=False):
    rows = load_manifest(f"{cfg['data']['manifest_dir']}/{split}.jsonl")
    t0 = time.time()
    hyps = decode_split(model, rows, cfg["eval"]["batch_size"], cfg["eval"]["max_new_tokens"])
    for r, h in zip(rows, hyps):
        r["hyp"] = h
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/hyps_{split}{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("utt_id", "speaker", "text", "hyp")}) + "\n")
    sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=model.device)
    res = evaluate(rows, group_key="speaker", sbert=sbert, bertscore=bertscore, judge=get_judge(cfg))
    res["decode_seconds"] = time.time() - t0
    json.dump(res, open(f"{out_dir}/metrics_{split}{tag}.json", "w"), indent=1)
    print(f"[eval {split}{tag}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in res["overall"].items()), flush=True)
    for r in rows[:3]:
        print(f"  REF: {r['text']}\n  HYP: {r['hyp']}")
    model.train()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", nargs="+", default=["test"]); ap.add_argument("--tag", default=""); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    feat_dim = np.load(feat_path(load_manifest(f"{cfg['data']['manifest_dir']}/dev.jsonl")[0])).shape[1]
    cfg["model"]["prune_vocab"] = os.path.exists(os.path.join(a.ckpt, "pruned_vocab"))   # match the checkpoint
    model = SpeechLLM(cfg, feat_dim, lora_path=a.ckpt, projector_path=f"{a.ckpt}/projector.pt").cuda()
    for s in a.splits:
        run_eval(model, cfg, s, os.path.dirname(a.ckpt.rstrip("/")), tag=a.tag, bertscore=cfg["eval"]["bertscore"])
