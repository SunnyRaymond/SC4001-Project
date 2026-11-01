# SC4001 Group Project (Direction I): When does few-shot CLIP adaptation hurt?

Code for the report. Frozen CLIP ViT-B/32 is adapted to each Office-Home domain with
K in {1, 2, 4, 8} labelled images per class. Each adapter's gain over zero-shot is split
into help (wrong to right) and harm (right to wrong), and per-sample confidence gating,
global mixing and a VRF-style weighting are compared on the same saved outputs.
Hyperparameters are selected only on the three source domains (leave-target-out);
target test labels are used once, for the final evaluation.

## Contents

```
configs/base.yaml        dataset/weight revisions, domains, prompts, K, seeds (fixed before any result)
configs/adapt.yaml       hyperparameter grids for A1, A2, A4 and gating (fixed before adapter results)
configs/vrf.yaml         few-shot VRF-style weighting: grid and hypothesis H3 (fixed before VRF results)
scripts/common.py        shared helpers: config, global image index, class names, cached features, splits
scripts/download.py      download the pinned Office-Home parquet and CLIP weights
scripts/make_splits.py   fixed nested support pools and test sets for every (domain, seed)
scripts/extract_features.py  frozen CLIP image/text features, once per backbone (GPU)
scripts/zero_shot.py     zero-shot baselines B0 (generic prompt) and B1 (domain prompt)
scripts/run_experiments.py   adapters A1, A2, A4 with leave-target-out selection; per-image outputs
scripts/analyze.py       help/harm decomposition, confidence bins, hypothesis H1
scripts/gating.py        confidence gates G1/G2, controls C1 (global mix), C2 (random gate), C3 (oracle); H2
scripts/vrf.py           VRF-style sample-wise weighting and the random-failure-set control V0; H3
scripts/vrf_signal.py    AUROC of MSP vs. failure-set distance for detecting B0 errors
scripts/plot.py          figures 1-4 and 6, and the K=4 main table
scripts/plot_examples.py figure 5 (harmed examples) and per-class harm; needs the images
report/build_figures.py  report figures (vector PDF) from the result CSVs
report/build_tables.py   report LaTeX tables from the result CSVs
jobs/                    environment setup and SLURM scripts used on the CCDS TC1 cluster
splits/                  the fixed splits used for every result (splits.npz, class_counts.csv)
results/ViT-B-32-quickgelu/  per-image predictions and summary tables behind the report
figures/ViT-B-32-quickgelu/  figures 1-6 (PNG) produced by plot.py and plot_examples.py
report/figures/, report/tables/  report figures and tables produced by report/build_*.py
requirements.txt, requirements-lock.txt  dependencies; the lock file lists the exact versions used
```

Docstrings and config comments cite sections of the team's design document as
"README x.y". That document is not part of this archive; the same design is described
in the report. README 3.1 (data, splits, leave-target-out selection) corresponds to report
Section 4; README 3.3 and 3.4 (methods, controls, metrics) to Section 3; README 3.6
(optional extensions: VRF-style weighting, ViT-B/16 check) to Section 5.4; README Section 7
(figures and tables) to Section 5.

## Existing code, models and data

- CLIP ViT-B/32 with OpenAI weights (Radford et al., 2021), loaded through
  `open_clip_torch==3.3.0` as `ViT-B-32-quickgelu` with weights from the Hugging Face repo
  `timm/vit_base_patch32_clip_224.openai`, revision pinned in `configs/base.yaml`.
  The model is used frozen; no CLIP code is modified.
- Office-Home (Venkateswara et al., 2017) from the Hugging Face mirror `flwrlabs/office-home`,
  revision pinned in `configs/base.yaml`. The images are for non-commercial research and
  teaching use only and are not included here.
- A1 uses scikit-learn's `LogisticRegression`. A2 (zero-shot-initialised linear probe),
  A4 (Tip-Adapter, Zhang et al., 2022), the gates, controls, VRF-style weighting
  (adapted from Zhu et al., 2024) and all analysis code were written by the team on top of
  NumPy, PyTorch, pandas and matplotlib.

## Dependencies

Python 3.11 was used. Install PyTorch first, then the rest:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

For a CPU-only machine use `--index-url https://download.pytorch.org/whl/cpu`.
A GPU is needed only for feature extraction (75 s on one V100; CPU also works but is slower).
Every later step runs on cached 512-d features on a CPU. `requirements-lock.txt` records the
exact package versions used on the TC1 cluster.

## Data preparation

```bash
python scripts/download.py       # Office-Home parquet (~1.16 GB) -> data/office_home; CLIP weights -> HF cache
python scripts/make_splits.py    # -> splits/splits.npz (already included; regenerating gives the same splits)
python scripts/extract_features.py --backbone ViT-B-32-quickgelu   # -> features/ViT-B-32-quickgelu/
```

The splits use seeds 0-4. For each (target domain, seed), the first 8 shuffled images per
class form a nested support pool (K-shot = the first K of the pool); every other image of
that domain is the test set, shared by all K and all methods.

## Reproducing the main results

Run from the archive root, in this order. Times are for a laptop CPU.

```bash
python scripts/zero_shot.py         # B0, B1                         (seconds)
python scripts/run_experiments.py   # A1, A2, A4 with selection      (~7 min)
python scripts/analyze.py           # help/harm, confidence bins, H1
python scripts/gating.py            # G1, G2, C1-C3, H2              (~1 min)
python scripts/vrf.py               # VRF-style weighting, V0, H3
python scripts/vrf_signal.py        # signal AUROC
python scripts/plot.py              # figures 1-4 and 6, main_table_k4.csv
python report/build_figures.py      # report/figures/*.pdf
python report/build_tables.py       # report/tables/*.tex
python scripts/plot_examples.py     # figure 5; needs data/office_home
```

All scripts default to `--backbone ViT-B-32-quickgelu` and write to `results/<backbone>/`
and `figures/<backbone>/`. `run_experiments.py` also writes `adapt_probs.npz` (~85 MB, the
full adapter probabilities), which `gating.py`, `vrf.py` and `vrf_signal.py` read; it is
not included in this archive because of its size. The other outputs are included, so the
analysis, plotting and report scripts can be run directly on them.

On the TC1 cluster the same steps were submitted as SLURM jobs (`jobs/setup_env.sh`, then
`sbatch jobs/download.sbatch`, `sbatch jobs/extract_features.sbatch`,
`sbatch jobs/zero_shot.sbatch`; `mkdir -p logs` first).

### Expected K=4 results (`results/ViT-B-32-quickgelu/main_table_k4.csv`)

Top-1 accuracy and harm (% of test images), mean over five seeds and four target domains:

| Method | Accuracy | Harm |
| --- | --- | --- |
| B0 zero-shot | 79.63 | 0.00 |
| A1 linear probe | 73.13 | 13.77 |
| A1 + confidence gate G1 | 80.39 | 3.24 |
| A1 + global mix C1 | 82.63 | 1.89 |
| A2 zero-shot-initialised probe | 82.99 | 3.74 |
| A2 + G1 | 82.97 | 3.53 |
| A2 + C1 | 83.24 | 2.43 |
| A4 Tip-Adapter | 81.44 | 1.95 |
| A4 + G1 | 81.49 | 1.87 |
| A4 + C1 | 81.42 | 1.89 |

The H1 check is in `h1.csv`; paired differences over all (domain, K, seed) runs for the
gates (H2) and the VRF-style weighting (H3) are in `gating_summary.csv` and `vrf_summary.csv`.
