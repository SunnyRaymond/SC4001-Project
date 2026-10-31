#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --job-name=oh_examples
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Figure 5 (harmed examples) needs the image bytes, which live only on TC1. CPU only.
# Needs results/<backbone>/{zero_shot_per_image.csv, adapt_per_image.npz} synced from the local copy.
# Usage (from the repo root): sbatch jobs/plot_examples.sbatch [extra args for plot_examples.py]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/envs/sc4001/bin/activate

python scripts/plot_examples.py "$@"
