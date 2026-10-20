"""Stage 2: GRPO on top of the SFT checkpoint with a configurable reward mixture.

Self-contained GRPO loop over *embedding* prompts (text + projected audio + text). One on-policy update per sampled group,
so the PPO ratio is 1 and the loss reduces to -A * logp + beta * KL(policy || SFT reference).
The reference is the SFT checkpoint itself: its LoRA is loaded a second time as a frozen adapter named
"ref" and its projector is kept as a frozen copy, so KL is exactly 0 at step 1.
"""
import argparse, copy, json, os, random, sys, time
import numpy as np, torch, yaml
from src.data.normalize import normalize
from src.model.speech_llm import SpeechLLM, load_manifest, load_feats, feat_path
from src.rewards.registry import RewardMixture
from src.eval.decode import run_eval


def main(cfg):
    if cfg["model"].get("prune_vocab") == "grpo":
        cfg["model"]["prune_vocab"] = True
    torch.manual_seed(0); random.seed(0)
    g = cfg["grpo"]
    train = load_manifest(f"{cfg['data']['manifest_dir']}/train.jsonl")
    feat_dim = np.load(feat_path(train[0])).shape[1]
    model = SpeechLLM(cfg, feat_dim, lora_path=g["init_from"], projector_path=f"{g['init_from']}/projector.pt").cuda()
    # frozen SFT reference: second (non-trainable) copy of the LoRA + projector
    model.llm.load_adapter(g["init_from"], adapter_name="ref")
    model.llm.set_adapter("default")
    for n, p in model.llm.named_parameters():
        p.requires_grad_(("lora_" in n) and (".ref." not in n))
    ref_projector = copy.deepcopy(model.projector).requires_grad_(False)
    reward_fn = RewardMixture(g["rewards"])
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=g["lr"], weight_decay=0.0)
    os.makedirs(g["out_dir"], exist_ok=True)
    log = open(f"{g['out_dir']}/train_log.jsonl", "a")
    run_eval(model, cfg, "dev", g["out_dir"], tag="_step0")
    G, t0, best = g["group_size"], time.time(), 1e9
    for step in range(1, g["steps"] + 1):
        batch = random.sample(train, g["utts_per_step"])
        feats, refs = load_feats(batch), [r["text"] for r in batch]
        # 1) sample a group of G completions per utterance
        model.eval()
        comp = model.generate(feats, max_new_tokens=g["max_new_tokens"], do_sample=True, temperature=g["temperature"], num_return_sequences=G)
        model.train()
        comps = []
        for c in comp:
            c = c.tolist()
            if model.eos_id in c:
                c = c[:c.index(model.eos_id) + 1]
            comps.append(torch.tensor(c, device=model.device))
        hyps = [normalize(model.decode(c)) for c in comps]
        # 2) rewards and group-normalized advantages
        rewards, comps_r = reward_fn(hyps, [r for r in refs for _ in range(G)])
        R = torch.tensor(rewards, device=model.device).view(len(batch), G)
        adv = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-4)).view(-1)
        # 3) policy gradient with KL to the frozen SFT reference (adapter "ref" + ref_projector)
        prompts = [p for p in model.prompt_embeds(feats) for _ in range(G)]
        lp, mask = model.completion_logprobs(prompts, comps)
        with torch.no_grad():
            proj, model.projector = model.projector, ref_projector
            model.llm.set_adapter("ref")
            lp_ref, _ = model.completion_logprobs([p for p in model.prompt_embeds(feats) for _ in range(G)], comps, chunk=4)
            model.llm.set_adapter("default"); model.projector = proj
            for n, p in model.llm.named_parameters():      # set_adapter toggles requires_grad; restore
                p.requires_grad_(("lora_" in n) and (".ref." not in n))
        kl = (torch.exp(lp_ref - lp) - (lp_ref - lp) - 1) * mask          # k3 estimator, per token
        per_tok = -(adv.unsqueeze(1) * lp) + g["kl_beta"] * kl
        loss = (per_tok.sum(1) / mask.sum(1).clamp(min=1)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        rec = {"step": step, "loss": loss.item(), "reward": R.mean().item(), "reward_std_in_group": R.std(1).mean().item(),
               "kl": (kl.sum() / mask.sum()).item(), "mean_len": mask.sum(1).float().mean().item(), "elapsed": time.time() - t0,
               "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9, **{f"r_{k}": float(np.mean(v)) for k, v in comps_r.items()}}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if step % 10 == 0:
            print(f"  REF: {refs[0]}\n  BEST: {hyps[int(R[0].argmax())]}\n  WORST: {hyps[int(R[0].argmin())]}", flush=True)
        if step % g["eval_every"] == 0 or step == g["steps"]:
            res = run_eval(model, cfg, "dev", g["out_dir"], tag=f"_step{step}")
            if res["overall"]["wer"] < best:
                best = res["overall"]["wer"]; model.save(f"{g['out_dir']}/best")
    model.save(f"{g['out_dir']}/last")
    print(f"GRPO done: best dev WER {best:.4f}, {time.time()-t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    main(yaml.safe_load(open(a.config)))
