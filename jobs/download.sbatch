#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=oh_download
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Download the pinned dataset and CLIP weights, then build the fixed splits.
# No GPU: compute nodes have internet (checked 2026-10-31). Submit from the repo root.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/envs/sc4001/bin/activate

python scripts/download.py
python scripts/make_splits.py
