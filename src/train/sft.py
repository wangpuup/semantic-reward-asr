"""Stage 1: supervised fine-tuning of projector + LoRA on cached features."""
import argparse, json, math, os, random, sys, time
import numpy as np, torch, yaml
from src.model.speech_llm import SpeechLLM, load_manifest, load_feats, feat_path
from src.eval.decode import run_eval


def main(cfg):
    if cfg["model"].get("prune_vocab") == "grpo":   # prune only at the GRPO stage
        cfg["model"]["prune_vocab"] = False
    torch.manual_seed(0); random.seed(0)
    s = cfg["sft"]
    train = load_manifest(f"{cfg['data']['manifest_dir']}/train.jsonl")
    feat_dim = np.load(feat_path(train[0])).shape[1]
    model = SpeechLLM(cfg, feat_dim).cuda()
    model.train()   # from_pretrained returns eval mode: without this, gradient checkpointing (and LoRA dropout) are off until the first run_eval
    model.llm.print_trainable_parameters()
    n_proj = sum(p.numel() for p in model.projector.parameters())
    print(f"projector params: {n_proj/1e6:.1f}M | GPU: {torch.cuda.get_device_name()}", flush=True)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=s["lr"], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / s["batch_size"])
    total = steps_per_epoch * s["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / s["warmup_steps"]) * max(0.0, (total - st) / total))
    best, step, t0 = 1e9, 0, time.time()
    os.makedirs(s["out_dir"], exist_ok=True)
    log = open(f"{s['out_dir']}/train_log.jsonl", "a")
    for ep in range(s["epochs"]):
        random.shuffle(train)
        # sort within chunks by length for less padding
        for i in range(0, len(train), s["batch_size"]):
            batch = train[i:i + s["batch_size"]]
            loss = model.sft_loss(load_feats(batch), [r["text"] for r in batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 20 == 0:
                rec = {"step": step, "epoch": ep, "loss": loss.item(), "lr": sched.get_last_lr()[0], "elapsed": time.time() - t0,
                       "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
                print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            if step % s["eval_every"] == 0 or step == total:
                res = run_eval(model, cfg, "dev", s["out_dir"], tag=f"_step{step}")
                wer = res["overall"]["wer"]
                if wer < best:
                    best = wer; model.save(f"{s['out_dir']}/best"); print(f"  new best dev WER {wer:.4f} -> saved", flush=True)
    model.save(f"{s['out_dir']}/last")
    print(f"SFT done: best dev WER {best:.4f}, {time.time()-t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    main(yaml.safe_load(open(a.config)))
