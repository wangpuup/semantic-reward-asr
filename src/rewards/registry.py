"""Reward functions. Each returns a list[float] for (hyps, refs); all texts are pre-normalized."""
import re
from collections import Counter
import jiwer, torch


class WERReward:
    """R_wer = 1 - min(WER, 1)."""
    def __init__(self, **kw): pass
    def __call__(self, hyps, refs):
        return [1.0 - min(jiwer.wer(r, h) if h.strip() else 1.0, 1.0) for h, r in zip(hyps, refs)]


class SBERTReward:
    """R_sem = cos(SBERT(hyp), SBERT(ref)) with a frozen encoder from a different family than the policy."""
    def __init__(self, model="sentence-transformers/all-mpnet-base-v2", **kw):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model, device="cuda" if torch.cuda.is_available() else "cpu")
    @torch.no_grad()
    def __call__(self, hyps, refs):
        e = self.m.encode(hyps + refs, convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
        n = len(hyps)
        return (e[:n] * e[n:]).sum(-1).clamp(0, 1).tolist()


class HallucinationReward:
    """R_hal = -1 for degenerate outputs (empty, length ratio out of range, repetition loops), else 0."""
    def __init__(self, min_ratio=0.5, max_ratio=2.0, ngram=4, max_repeat=3, **kw):
        self.min_ratio, self.max_ratio, self.n, self.max_repeat = min_ratio, max_ratio, ngram, max_repeat
    def __call__(self, hyps, refs):
        out = []
        for h, r in zip(hyps, refs):
            hw, rw = h.split(), r.split()
            bad = len(hw) == 0 or not (self.min_ratio <= len(hw) / max(len(rw), 1) <= self.max_ratio)
            if not bad and len(hw) >= self.n:
                grams = Counter(tuple(hw[i:i + self.n]) for i in range(len(hw) - self.n + 1))
                bad = max(grams.values()) > self.max_repeat
            out.append(-1.0 if bad else 0.0)
        return out


class KeywordReward:
    """Content-word F1 (stopwords removed), FunAudio-ASR style."""
    STOP = set("a an the and or but of to in on at for with by from as is are was were be been being it its this that these those i you he she we they me him her us them my your his our their not no yes do does did have has had".split())
    def __init__(self, **kw): pass
    def __call__(self, hyps, refs):
        out = []
        for h, r in zip(hyps, refs):
            hc, rc = Counter(w for w in h.split() if w not in self.STOP), Counter(w for w in r.split() if w not in self.STOP)
            tp = sum((hc & rc).values())
            p, q = tp / max(sum(hc.values()), 1), tp / max(sum(rc.values()), 1)
            out.append(0.0 if tp == 0 else 2 * p * q / (p + q))
        return out


REWARDS = {"wer": WERReward, "sbert": SBERTReward, "hallucination": HallucinationReward, "keyword": KeywordReward}


class RewardMixture:
    """Weighted sum of named rewards; also returns the per-component values for logging."""
    def __init__(self, specs):
        self.parts = [(s["name"], float(s.get("weight", 1.0)), REWARDS[s["name"]](**{k: v for k, v in s.items() if k not in ("name", "weight")})) for s in specs]
    def __call__(self, hyps, refs):
        comps = {name: fn(hyps, refs) for name, _, fn in self.parts}
        total = [sum(w * comps[name][i] for name, w, _ in self.parts) for i in range(len(hyps))]
        return total, comps


class BERTScoreReward:
    """R_bs = BERTScore F1 (roberta-large), rescaled with baseline so ~0..1."""
    def __init__(self, model_type="roberta-large", **kw):
        from bert_score import BERTScorer
        self.scorer = BERTScorer(model_type=model_type, lang="en", rescale_with_baseline=True,
                                 device="cuda" if torch.cuda.is_available() else "cpu")
    @torch.no_grad()
    def __call__(self, hyps, refs):
        _, _, f1 = self.scorer.score(hyps, refs)
        return f1.clamp(0, 1).tolist()


class LLMJudgeReward:
    """R_judge = P(yes) of a frozen instruction-tuned LLM asked whether the hypothesis preserves the
    meaning of the reference. One forward pass per pair (no generation): reward = p(yes)/(p(yes)+p(no))
    read from the next-token distribution. Keep the judge a *different* model than the policy."""
    PROMPT = ("Reference transcript: \"{ref}\"\nASR hypothesis: \"{hyp}\"\n\n"
              "Does the hypothesis preserve the meaning of the reference, ignoring spelling, punctuation and "
              "minor wording differences that do not change what is said? Answer with yes or no.")

    def __init__(self, model="Qwen/Qwen2.5-1.5B-Instruct", batch_size=16, **kw):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).cuda().eval()
        self.bs = batch_size
        ids = lambda words: sorted({self.tok.encode(w, add_special_tokens=False)[0] for w in words})
        self.yes_ids, self.no_ids = ids(["yes", "Yes", " yes", " Yes", "YES"]), ids(["no", "No", " no", " No", "NO"])

    def _chat(self, hyp, ref):
        msgs = [{"role": "system", "content": "You are a strict but fair judge of speech recognition output."},
                {"role": "user", "content": self.PROMPT.format(ref=ref, hyp=hyp)}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def __call__(self, hyps, refs):
        out = []
        for i in range(0, len(hyps), self.bs):
            texts = [self._chat(h if h.strip() else "(empty)", r) for h, r in zip(hyps[i:i + self.bs], refs[i:i + self.bs])]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.m.device)
            logits = self.m(**enc).logits[:, -1].float()
            p = torch.softmax(logits, -1)
            py, pn = p[:, self.yes_ids].sum(-1), p[:, self.no_ids].sum(-1)
            out += (py / (py + pn + 1e-9)).tolist()
        return out


REWARDS.update({"bertscore": BERTScoreReward, "llm_judge": LLMJudgeReward})
