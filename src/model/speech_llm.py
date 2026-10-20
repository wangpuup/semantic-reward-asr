"""Cached-encoder-features -> projector -> LoRA LLM (SLAM-LLM style, continuous features)."""
import json, os
import numpy as np, torch, torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class Projector(nn.Module):
    def __init__(self, feat_dim, hidden, llm_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU(), nn.Linear(hidden, llm_dim))

    def forward(self, x):
        return self.net(x)


class SpeechLLM(nn.Module):
    def __init__(self, cfg, feat_dim, lora_path=None, projector_path=None):
        super().__init__()
        m = cfg["model"]
        self.tok = AutoTokenizer.from_pretrained(m["llm"])
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # prompt pieces (no chat template: plain SLAM-LLM style prompt works for base and instruct models)
        self.pre_ids = self.tok("USER: ", add_special_tokens=False).input_ids
        self.post_ids = self.tok(f" {m['prompt']}\nASSISTANT:", add_special_tokens=False).input_ids
        self.eos_id = self.tok.eos_token_id
        self.pad_id = self.tok.pad_token_id
        self.o2n, self.n2o = None, None            # old<->new token-id maps when the vocab is pruned
        llm = AutoModelForCausalLM.from_pretrained(m["llm"], torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        llm.config.use_cache = True
        if m.get("prune_vocab"):
            self._prune_vocab(llm, cfg)
        if lora_path:
            self.llm = PeftModel.from_pretrained(llm, lora_path, is_trainable=True)
        else:
            self.llm = get_peft_model(llm, LoraConfig(r=m["lora_r"], lora_alpha=m["lora_alpha"], lora_dropout=m["lora_dropout"],
                                                       target_modules=m["lora_targets"], task_type="CAUSAL_LM"))
        if m.get("grad_checkpoint"):
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.projector = Projector(feat_dim, m["projector_hidden"], llm.config.hidden_size).to(torch.bfloat16)
        if projector_path:
            self.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))

    # ---- helpers -------------------------------------------------------------------------------
    def _prune_vocab(self, llm, cfg):
        """Slice the embedding/LM head down to the token ids that can actually occur (corpus text +
        prompt + specials). The (B, L, vocab) logits dominate SFT/GRPO memory, so a 151k -> ~10k vocab
        cuts that term ~15x. Deterministic given manifests+tokenizer, so re-applied identically when a
        LoRA checkpoint from a pruned run is loaded back."""
        keep = set(self.pre_ids + self.post_ids + [self.eos_id, self.pad_id])
        mdir = cfg["data"]["manifest_dir"]
        for split in ("train", "dev", "test"):
            path = os.path.join(mdir, f"{split}.jsonl")
            if os.path.exists(path):
                for l in open(path):
                    t = json.loads(l)["text"]
                    keep.update(self.tok(t, add_special_tokens=False).input_ids)
                    keep.update(self.tok(" " + t, add_special_tokens=False).input_ids)
        keep = sorted(keep)
        idx = torch.tensor(keep)
        emb = llm.get_input_embeddings()
        new_emb = nn.Embedding(len(keep), emb.weight.shape[1], dtype=emb.weight.dtype)
        new_emb.weight.data.copy_(emb.weight.data[idx])
        llm.set_input_embeddings(new_emb)
        if not getattr(llm.config, "tie_word_embeddings", False):
            head = llm.get_output_embeddings()
            new_head = nn.Linear(head.in_features, len(keep), bias=False, dtype=head.weight.dtype)
            new_head.weight.data.copy_(head.weight.data[idx])
            llm.set_output_embeddings(new_head)
        llm.config.vocab_size = len(keep)
        llm.tie_weights()
        self.o2n = {o: n for n, o in enumerate(keep)}
        self.n2o = keep
        self.pre_ids = [self.o2n[i] for i in self.pre_ids]
        self.post_ids = [self.o2n[i] for i in self.post_ids]
        self.eos_id, self.pad_id = self.o2n[self.eos_id], self.o2n[self.pad_id]
        print(f"vocab pruned: {emb.weight.shape[0]} -> {len(keep)}", flush=True)

    def encode(self, text):
        ids = self.tok(text, add_special_tokens=False).input_ids
        return [self.o2n[i] for i in ids] if self.o2n else ids

    def decode(self, ids):
        ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
        if self.n2o is not None:
            ids = [self.n2o[i] for i in ids]
        return self.tok.decode(ids, skip_special_tokens=True)

    @property
    def device(self):
        return next(self.llm.parameters()).device

    def embed_tokens(self, ids):
        return self.llm.get_input_embeddings()(torch.tensor(ids, device=self.device))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def prompt_embeds(self, feats):
        """feats: list of (T, D) fp16 arrays -> list of (L, H) prompt embeddings (text + audio + text)."""
        pre, post = self.embed_tokens(self.pre_ids), self.embed_tokens(self.post_ids)
        out = []
        for f in feats:
            a = self.projector(torch.as_tensor(f, device=self.device, dtype=torch.bfloat16))
            out.append(torch.cat([pre, a, post], 0))
        return out

    def build_batch(self, prompts, targets=None, pad_left=True):
        """Pack variable-length [prompt (+ target tokens)] into a padded batch.
        Returns inputs_embeds, attention_mask, labels (None if no targets), prompt lengths."""
        seqs, labels = [], []
        for i, p in enumerate(prompts):
            if targets is not None:
                t = torch.as_tensor(targets[i], device=self.device)
                seqs.append(torch.cat([p, self.llm.get_input_embeddings()(t)], 0))
                labels.append(torch.cat([torch.full((p.shape[0],), -100, device=self.device), t]))
            else:
                seqs.append(p)
        L = max(s.shape[0] for s in seqs)
        H = seqs[0].shape[1]
        emb = torch.zeros(len(seqs), L, H, device=self.device, dtype=torch.bfloat16)
        att = torch.zeros(len(seqs), L, device=self.device, dtype=torch.long)
        lab = torch.full((len(seqs), L), -100, device=self.device, dtype=torch.long) if targets is not None else None
        for i, s in enumerate(seqs):
            n = s.shape[0]
            sl = slice(L - n, L) if pad_left else slice(0, n)
            emb[i, sl], att[i, sl] = s, 1
            if lab is not None:
                lab[i, sl] = labels[i]
        return emb, att, lab

    # ---- training / inference ------------------------------------------------------------------
    def sft_loss(self, feats, texts, chunk_tokens=1024):
        """Cross-entropy without materialising the full (B, L, vocab) logits: run the transformer once,
        then apply the LM head + loss over flat slices of `chunk_tokens` label positions. Saves the
        ~8-10 GB logit tensor that dominates full-vocab SFT memory (gradient checkpointing can't touch
        it). Gives the same loss as passing labels= to the model."""
        targets = [self.encode(t) + [self.eos_id] for t in texts]
        emb, att, lab = self.build_batch(self.prompt_embeds(feats), targets, pad_left=False)
        base = self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm
        hidden = base.model(inputs_embeds=emb, attention_mask=att).last_hidden_state[:, :-1]
        tgt = lab[:, 1:]
        mask = (tgt != -100).reshape(-1)
        flat_h, flat_t = hidden.reshape(-1, hidden.shape[-1])[mask], tgt.reshape(-1)[mask]
        head = base.get_output_embeddings()
        def chunk_ce(h, t):
            return torch.nn.functional.cross_entropy(head(h).float(), t, reduction="sum")
        losses = []
        for i in range(0, flat_h.shape[0], chunk_tokens):
            h, t = flat_h[i:i + chunk_tokens], flat_t[i:i + chunk_tokens]
            # checkpoint: the chunk's (chunk, vocab) fp32 logits are recomputed in backward instead of
            # stored, so only one chunk's logits are ever alive - without this, autograd keeps them all
            # and chunking saves nothing
            losses.append(torch.utils.checkpoint.checkpoint(chunk_ce, h, t, use_reentrant=False)
                          if torch.is_grad_enabled() and flat_h.requires_grad else chunk_ce(h, t))
        return torch.stack(losses).sum() / mask.sum()

    @torch.no_grad()
    def generate(self, feats, max_new_tokens=128, do_sample=False, temperature=1.0, num_return_sequences=1):
        emb, att, _ = self.build_batch(self.prompt_embeds(feats), pad_left=True)
        out = self.llm.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=max_new_tokens,
                                do_sample=do_sample, temperature=temperature if do_sample else None, top_p=None, top_k=None,
                                num_return_sequences=num_return_sequences, pad_token_id=self.pad_id, eos_token_id=self.eos_id)
        # with inputs_embeds, generate() returns only the new tokens
        return out

    def completion_logprobs(self, prompts, completions, chunk=None):
        """Per-token log p(completion | prompt). completions: list of 1-D LongTensors.
        Memory note: the (B, L, vocab) logits dominate GRPO memory (150k vocab). `chunk` runs the batch in
        slices of that many sequences (use it for the no-grad reference pass); cross_entropy avoids
        materialising a second full-size log-softmax tensor."""
        if chunk and len(prompts) > chunk:
            outs = [self.completion_logprobs(prompts[i:i + chunk], completions[i:i + chunk]) for i in range(0, len(prompts), chunk)]
            L = max(o[0].shape[1] for o in outs)
            pad = lambda t: torch.nn.functional.pad(t, (L - t.shape[1], 0))   # left-padded batches -> pad on the left
            return torch.cat([pad(o[0]) for o in outs]), torch.cat([pad(o[1]) for o in outs])
        emb, att, lab = self.build_batch(prompts, [c.tolist() for c in completions], pad_left=True)
        logits = self.llm(inputs_embeds=emb, attention_mask=att).logits[:, :-1]
        tgt = lab[:, 1:]
        mask = tgt != -100
        lp = -torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), tgt.clamp(min=0), reduction="none")
        return lp * mask, mask

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.llm.save_pretrained(out_dir)
        torch.save(self.projector.state_dict(), os.path.join(out_dir, "projector.pt"))
        if self.o2n is not None:   # marker so eval reloads this checkpoint with the pruned vocab
            open(os.path.join(out_dir, "pruned_vocab"), "w").close()


def load_manifest(path):
    return [json.loads(l) for l in open(path)]


def feat_path(r):
    """Manifest paths point at the NFS cache; FEATURE_DIR (set by run_demo.sh under Condor) redirects
    to the copy that HTCondor transferred into the job's local scratch directory."""
    root = os.environ.get("FEATURE_DIR")
    return os.path.join(root, r["split"], r["utt_id"] + ".npy") if root else r["feat"]


def load_feats(rows):
    return [np.load(feat_path(r)) for r in rows]
