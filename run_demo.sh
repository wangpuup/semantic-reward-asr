#!/bin/bash
# Usage: run_demo.sh <stage|all|prep|train|rl> [config]
#   stages: manifest features sft grpo eval
#   prep  = manifest + features (writes the feature cache once)
#   train = sft + grpo + eval;  rl = grpo + eval
# Run from anywhere; under HTCondor the job files pass PROJECT_ROOT (the submit directory).
# With ShouldTransferFiles, train.job copies data/features into the job's scratch dir and
# this script points FEATURE_DIR at that copy so training never reads the shared filesystem.
set -euo pipefail
STAGE=${1:-all}; CFG=${2:-configs/example.yaml}
cd "${PROJECT_ROOT:-$(dirname "$(readlink -f "$0")")}"   # PROJECT_ROOT is set by the Condor job files (submit dir)
CONDA_ENV=${CONDA_ENV:-rlasr}
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"
export HF_HOME=${HF_HOME:-$PWD/hf_cache} TOKENIZERS_PARALLELISM=false PYTHONPATH=$PWD
if [[ -n "${_CONDOR_SCRATCH_DIR:-}" && -d "$_CONDOR_SCRATCH_DIR/features" ]]; then
  export FEATURE_DIR=$_CONDOR_SCRATCH_DIR/features; echo "features staged in local scratch: $FEATURE_DIR"
fi
echo "host=$(hostname) date=$(date) stage=$STAGE cfg=$CFG"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'"
run() { echo "=== $1 ($(date +%H:%M:%S)) ==="; shift; "$@"; }
cfgval() { python -c "import yaml,sys;print(yaml.safe_load(open('$CFG'))$1)"; }
[[ $STAGE == all || $STAGE == prep  || $STAGE == manifest ]] && run manifest python -m src.data.manifest   --config "$CFG"
[[ $STAGE == all || $STAGE == prep  || $STAGE == features ]] && run features python -m src.features.extract --config "$CFG"
[[ $STAGE == all || $STAGE == train || $STAGE == sft      ]] && run sft      python -m src.train.sft        --config "$CFG"
[[ $STAGE == all || $STAGE == train || $STAGE == rl || $STAGE == grpo ]] && run grpo python -m src.train.grpo --config "$CFG"
if [[ $STAGE == all || $STAGE == train || $STAGE == eval ]]; then
  run eval_sft  python -m src.eval.decode --config "$CFG" --ckpt "$(cfgval "['sft']['out_dir']")/best"  --splits test --tag _sft
fi
if [[ $STAGE == all || $STAGE == train || $STAGE == rl || $STAGE == eval ]]; then
  run eval_grpo python -m src.eval.decode --config "$CFG" --ckpt "$(cfgval "['grpo']['out_dir']")/best" --splits test --tag _grpo
fi
echo "=== done $(date) ==="
