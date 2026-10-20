#!/bin/bash
# Creates the `rlasr` conda env used by all stages (CUDA 12.1 wheels; adjust the torch index for your driver).
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n rlasr python=3.11
conda activate rlasr
pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir "transformers==4.51.3" "peft==0.15.2" "accelerate==1.6.0" \
  "sentence-transformers==3.4.1" "bert-score==0.3.13" "jiwer==3.1.0" kaldiio soundfile numpy pyyaml tqdm
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'peft', peft.__version__)"
echo ENV_OK
