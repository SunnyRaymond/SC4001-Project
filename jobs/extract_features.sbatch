#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=oh_feats
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage (from the repo root): sbatch jobs/extract_features.sbatch [BACKBONE]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/envs/sc4001/bin/activate
export HF_HUB_OFFLINE=1  # weights were fetched by download.sbatch
BACKBONE=${1:-ViT-B-32-quickgelu}

# Log the allocated GPU's utilisation and memory every 60 s.
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 60 > "logs/gpu_${SLURM_JOB_ID}.csv" &
SMI=$!
trap 'kill $SMI' EXIT

python scripts/extract_features.py --backbone "$BACKBONE" --workers "$SLURM_CPUS_PER_TASK"
