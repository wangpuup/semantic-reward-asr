"""WER / CER / semantic metrics with optional per-group breakdown. Inputs are normalized text."""
import json
from collections import defaultdict
import jiwer, torch


def corpus_scores(hyps, refs, sbert=None, bertscore=False, judge=None):
    out = {"n": len(refs), "wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps)}
    if judge is not None:
        p = judge(hyps, refs)
        out["judge_p_yes"] = sum(p) / len(p)
        out["judge_preserved_rate"] = sum(x > 0.5 for x in p) / len(p)
    if sbert is not None:
        e = sbert.encode(hyps + refs, convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
        out["sbert_cos"] = (e[:len(hyps)] * e[len(hyps):]).sum(-1).mean().item()
    if bertscore:
        from bert_score import score
        _, _, f1 = score(hyps, refs, model_type="roberta-large", lang="en", verbose=False)
        out["bertscore_f1"] = f1.mean().item()
    return out


def evaluate(rows, group_key=None, sbert=None, bertscore=False, judge=None):
    """rows: dicts with 'hyp','text' (+ group_key). Returns overall and per-group metrics."""
    hyps, refs = [r["hyp"] if r["hyp"].strip() else "<empty>" for r in rows], [r["text"] for r in rows]
    res = {"overall": corpus_scores(hyps, refs, sbert, bertscore, judge)}
    if group_key:
        groups = defaultdict(list)
        for r in rows:
            groups[r.get(group_key, "unk")].append(r)
        res["by_" + group_key] = {g: corpus_scores([r["hyp"] or "<empty>" for r in rs], [r["text"] for r in rs], sbert, bertscore, judge) for g, rs in sorted(groups.items())}
    return res
