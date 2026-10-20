# Semantic rewards for LLM-based ASR — reference pipeline

A small, self-contained pipeline for the thesis *"Underrepresented Speech Recognition with LLMs Using
Reinforcement Learning: Semantic Correctness as a Reward"*:

frozen speech encoder → trained linear projector → LLM with LoRA → **supervised fine-tuning** →
**GRPO** with pluggable (semantic) rewards → per-group evaluation.

## Layout
```
src/data/       manifests from an ESPnet-style dump/raw recipe; one shared text normalizer
src/features/   frozen Whisper encoder -> stacked fp16 features cached as .npy
src/model/      projector + LoRA-LLM: prompt building, SFT loss, decoding (greedy / sampling), vocab pruning
src/rewards/    reward registry: wer, keyword, sbert, bertscore, llm_judge, hallucination, weighted mixtures
src/train/      sft.py (supervised fine-tuning) and grpo.py (GRPO loop)
src/eval/       greedy decoding, WER/CER/semantic metrics, per-speaker/per-group breakdown
configs/        one experiment = one YAML; configs/example.yaml is the documented template
condor/         HTCondor submit files (prep.job writes the feature cache once; train.job transfers it to scratch)
run_demo.sh     stage runner: manifest | features | sft | grpo | eval  (or prep / train / rl / all)
```

## Setup
```bash
bash setup_env.sh                          # conda env `rlasr` (torch 2.5.1 + cu121, transformers 4.51.3, peft 0.15.2)
# edit configs/example.yaml: data.kaldi_root (required); optionally model.llm, grpo.rewards, out_dirs
mkdir -p logs
```

## Stages
`run_demo.sh <what> <config>` runs one or more of five stages, always in this order:

| stage | what it does | writes |
|---|---|---|
| `manifest` | list utterances from the ESPnet-style recipe (`data.kaldi_root`) | `data/manifests/{train,dev,test}.jsonl` |
| `features` | run the frozen encoder once, cache features | `data/features/<split>/<utt>.npy` |
| `sft` | supervised fine-tuning of projector + LoRA | `<sft.out_dir>/{best,last}/` |
| `grpo` | RL from the SFT checkpoint with the reward mixture | `<grpo.out_dir>/{best,last}/` |
| `eval` | greedy-decode the test set with both checkpoints | `metrics_test_*.json`, `hyps_test_*.jsonl` |

Bundles: `prep` = manifest + features · `train` = sft + grpo + eval · `rl` = grpo + eval · `all` = everything.

## Running on HTCondor (recommended)
Always submit from the repository root. Two job files are provided:

```bash
# 1. once per dataset: manifests + feature cache (written to the shared filesystem)
condor_submit condor/prep.job                       # = run_demo.sh prep configs/example.yaml

# 2. per experiment: SFT -> GRPO -> eval
condor_submit condor/train.job                      # = run_demo.sh train configs/example.yaml

# another config (e.g. a copy of example.yaml with a different reward list) and only GRPO + eval,
# reusing the SFT checkpoint named in its grpo.init_from:
condor_submit -a 'Arguments = rl configs/other.yaml' condor/train.job
```
`train.job` uses `ShouldTransferFiles = yes` and copies `data/features` + `data/manifests` into the node's
local scratch directory before training, so the training loop never reads the shared filesystem; `run_demo.sh`
detects the copy automatically. Logs stream to `logs/train_<cluster>.{out,err,log}` while the job runs
(`stream_output = true`). Edit `Requirements` in the job files for your cluster's GPU names / memory.

## Running locally, stage by stage
On any machine with a GPU and the `rlasr` env (e.g. an interactive GPU node):

```bash
./run_demo.sh manifest configs/example.yaml
./run_demo.sh features configs/example.yaml
./run_demo.sh sft      configs/example.yaml
./run_demo.sh grpo     configs/example.yaml
./run_demo.sh eval     configs/example.yaml
```
or any bundle, e.g. `./run_demo.sh all configs/example.yaml`. The script `cd`s to its own directory, so relative
paths in the config resolve against the repository root wherever you call it from. Each stage only needs the
outputs of the previous ones, so re-running a later stage (new reward, new learning rate) never repeats the
earlier work.

## Rewards
`grpo.rewards` is a weighted list; each entry names a class in `src/rewards/registry.py`:

| name | definition | note |
|---|---|---|
| `wer` | 1 − WER | orthographic baseline |
| `keyword` | content-word F1 (stop-words removed) | between spelling and meaning |
| `sbert` | cosine similarity of sentence embeddings | primary semantic reward |
| `bertscore` | BERTScore F1 | token-level semantic alternative |
| `llm_judge` | P(yes) of a frozen instruct LLM asked "same meaning?" (one forward pass) | +~3.5 GB, ~30 % wall time |
| `hallucination` | −1 on length blow-up or repeated n-grams | guard for the semantic rewards |

Adding a reward = one class with `__call__(hyps, refs) -> list[float]` registered in the same file.

## Memory settings
Two config flags make the whole 0.5B–7B range fit a 24 GB GPU (measured on 0.5B/1.7B/3B/7B):
`model.grad_checkpoint: true` and `model.prune_vocab: grpo` (LM head restricted to the tokens that occur
in the corpus, applied at the RL stage only).
