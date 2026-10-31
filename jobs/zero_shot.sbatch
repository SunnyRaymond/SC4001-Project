#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=oh_zeroshot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# B0/B1 on cached features; CPU only, takes seconds.
# Usage (from the repo root): sbatch jobs/zero_shot.sbatch [BACKBONE]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/envs/sc4001/bin/activate

python scripts/zero_shot.py --backbone "${1:-ViT-B-32-quickgelu}"
