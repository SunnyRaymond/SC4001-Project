#!/bin/bash
# One-time environment setup on the TC1 head node: bash jobs/setup_env.sh
#
# Why not the course conda env: /tc1apps/2_conda_env/CZ4042_v5 (PyTorch 2.5.1, CUDA 11.8)
# pins mkl=2025.0.0, and its torch fails to import with
# "libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent" (checked 2026-10-31).
# The official pip wheels bundle their own CUDA libraries and still support V100 (sm_70).
set -euo pipefail

ENV=~/envs/sc4001
module load python/3.11
python3.11 -m venv "$ENV"
module unload python/3.11
source "$ENV/bin/activate"

python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
# The constraint stops later packages from pulling in a newer torch.
python -m pip install --no-cache-dir -r "$(dirname "$0")/../requirements.txt" \
    -c <(printf "torch==2.5.1+cu118\ntorchvision==0.20.1+cu118\n")

python -c "import torch, open_clip; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'open_clip', open_clip.__version__)"
