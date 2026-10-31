"""Failure analysis: harmed images and the classes harmed most (README 3.4, figure 5).

An image is harmed when B0 is right and the adapter is wrong. For one adapter,
K and seed, a fixed random sample of harmed test images per domain is drawn and
shown with the true class, B0's confidence and the adapter's prediction.
Needs the Office-Home parquet shards (data/office_home), so it runs on TC1
(jobs/plot_examples.sbatch); the per-class table needs only the results.

Writes:
  figures/<backbone>/fig5_harmed_<M>_k<K>.png
  results/<backbone>/harm_by_class_k<K>.csv  per (method, domain, class), pooled over seeds
"""

import argparse
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

import common
import plot

N_PER_DOMAIN = 4


def read_images(data_dir, indices):
    """Decoded images for global indices (row order of the sorted parquet shards)."""
    wanted, out, offset = set(int(i) for i in indices), {}, 0
    for p in common.shard_paths(data_dir):
        n = pq.read_metadata(p).num_rows
        rows = [i - offset for i in wanted if offset <= i < offset + n]
        if rows:
            col = pq.read_table(p, columns=["image"]).column("image").combine_chunks().field("bytes")
            for r in rows:
                out[offset + r] = Image.open(io.BytesIO(col[r].as_py())).convert("RGB")
        offset += n
    return [out[int(i)] for i in indices]


def harm_by_class(zs, adapt, splits, domains, seeds, k, names):
    rows = []
    methods = sorted({key.split("/")[0] for key in adapt.files})
    for method in methods:
        for domain, slug in domains.items():
            parts = []
            for seed in seeds:
                test = splits[(slug, seed)][1]
                y = zs["label"].to_numpy()[test]
                b0_ok = zs["B0_pred"].to_numpy()[test] == y
                ad_ok = adapt[f"{method}/{slug}/K{k}/seed{seed}/pred"].astype(np.int64) == y
                parts.append(pd.DataFrame({"label": y, "b0_ok": b0_ok, "harm": b0_ok & ~ad_ok}))
            g = pd.concat(parts).groupby("label").agg(n=("b0_ok", "size"), b0_right=("b0_ok", "sum"),
                                                      harmed=("harm", "sum"))
            g["cond_harm"] = g.harmed / g.b0_right.clip(lower=1)
            g = g.reset_index().assign(method=method, domain=domain, class_name=lambda d: [names[c] for c in d.label])
            rows.append(g)
    cols = ["method", "domain", "label", "class_name", "n", "b0_right", "harmed", "cond_harm"]
    return pd.concat(rows)[cols]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=common.DEFAULT_CONFIG)
    ap.add_argument("--backbone", default="ViT-B-32-quickgelu")
    ap.add_argument("--method", default="A2")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="split seed of the shown images; also seeds the sample")
    ap.add_argument("--data", help="default: dataset.local_dir from the config")
    ap.add_argument("--results", default=common.REPO_ROOT / "results")
    ap.add_argument("--figures", default=common.REPO_ROOT / "figures")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    domains = cfg["dataset"]["domains"]
    data_dir = args.data or common.REPO_ROOT / cfg["dataset"]["local_dir"]
    res = common.feature_dir(args.backbone, args.results)
    zs = pd.read_csv(res / "zero_shot_per_image.csv", index_col="index")
    adapt = np.load(res / "adapt_per_image.npz")
    names = common.load_features(args.backbone)["info"]["class_names"]
    splits = common.load_splits()

    table = harm_by_class(zs, adapt, splits, domains, cfg["splits"]["seeds"], args.k, names)
    table.to_csv(res / f"harm_by_class_k{args.k}.csv", index=False, float_format="%.6f")
    print(f"{args.method}, K={args.k}: classes with the most harmed images (pooled over seeds)")
    for domain in domains:
        t = table[(table.method == args.method) & (table.domain == domain)].nlargest(5, "harmed")
        print(f"  {domain}: " + ", ".join(f"{r.class_name} {r.harmed}/{r.b0_right}" for r in t.itertuples()))

    rng = np.random.default_rng(args.seed)
    shown = []
    for domain, slug in domains.items():
        test = splits[(slug, args.seed)][1]
        y = zs["label"].to_numpy()[test]
        ad_pred = adapt[f"{args.method}/{slug}/K{args.k}/seed{args.seed}/pred"].astype(np.int64)
        harmed = np.flatnonzero((zs["B0_pred"].to_numpy()[test] == y) & (ad_pred != y))
        pick = np.sort(rng.choice(harmed, min(N_PER_DOMAIN, len(harmed)), replace=False))
        shown += [(domain, test[i], y[i], ad_pred[i], zs["B0_msp"].iloc[test[i]]) for i in pick]

    images = read_images(data_dir, [s[1] for s in shown])
    plot.style()
    fig, axes = plt.subplots(len(domains), N_PER_DOMAIN, figsize=(8, 2.35 * len(domains)))
    for ax, (domain, idx, y, pred, conf), img in zip(axes.flat, shown, images):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(f"true: {names[y]}", fontsize=8, color=plot.INK)
        ax.set_xlabel(f"B0 right, c(x)={conf:.2f}\n{args.method}: {names[pred]}", fontsize=7, color=plot.INK2)
    for ax, domain in zip(axes[:, 0], domains):
        ax.set_ylabel(domain, fontsize=9, color=plot.INK)
    fig.suptitle(f"Images B0 classifies correctly but {args.method} (K={args.k}) gets wrong; "
                 f"random sample, split seed {args.seed}", fontsize=9, color=plot.INK2)
    fig.tight_layout()
    out = common.feature_dir(args.backbone, args.figures)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"fig5_harmed_{args.method}_k{args.k}.png", dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
