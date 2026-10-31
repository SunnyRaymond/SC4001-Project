"""Zero-shot baselines B0 and B1 on cached CLIP features (README 3.3).

B0: "a photo of a {c}." for every domain.
B1: the fixed domain prompt of each image's own domain (configs/base.yaml).
Logits use CLIP's own logit scale; confidence c(x) is the max softmax probability.

Writes results/<backbone>/:
  zero_shot_per_image.csv  index, domain, label, <M>_pred, <M>_msp for M in B0, B1
  zero_shot_summary.csv    accuracy on the whole domain and on each seed's test set
"""

import argparse

import numpy as np
import pandas as pd

import common


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=common.DEFAULT_CONFIG)
    ap.add_argument("--backbone", default="ViT-B-32-quickgelu")
    ap.add_argument("--out", default=common.REPO_ROOT / "results")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    domains = cfg["dataset"]["domains"]
    feats = common.load_features(args.backbone)
    X, meta, text = feats["image"], feats["meta"], feats["text"]
    scale = feats["info"]["logit_scale"]
    splits = common.load_splits()

    per_image = meta[["domain", "label"]].copy()
    for method in ("B0", "B1"):
        probs = np.empty((len(X), cfg["dataset"]["num_classes"]), dtype=np.float32)
        for domain, slug in domains.items():
            rows = (meta["domain"] == domain).to_numpy()
            T = text["b0"] if method == "B0" else text[f"b1_{slug}"]
            probs[rows] = softmax(scale * X[rows] @ T.T)
        per_image[f"{method}_pred"] = probs.argmax(1)
        per_image[f"{method}_msp"] = probs.max(1)

    rows = []
    for method in ("B0", "B1"):
        correct = (per_image[f"{method}_pred"] == per_image["label"]).to_numpy()
        for domain, slug in domains.items():
            in_domain = (per_image["domain"] == domain).to_numpy()
            seed_acc = [correct[splits[(slug, s)][1]].mean() for s in cfg["splits"]["seeds"]]
            rows.append({
                "backbone": args.backbone, "method": method, "domain": domain,
                "acc_all": correct[in_domain].mean(),
                "acc_test_mean": np.mean(seed_acc), "acc_test_std": np.std(seed_acc, ddof=1),
                **{f"acc_test_seed{s}": a for s, a in zip(cfg["splits"]["seeds"], seed_acc)},
            })
    summary = pd.DataFrame(rows)

    out = common.feature_dir(args.backbone, args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(out / "zero_shot_per_image.csv", float_format="%.6f")
    summary.to_csv(out / "zero_shot_summary.csv", index=False, float_format="%.6f")

    # Test-set accuracy (mean ± std over seeds) in percent, one column per domain.
    table = summary.assign(cell=lambda d: [f"{m * 100:.2f} ± {s * 100:.2f}" for m, s in zip(d.acc_test_mean, d.acc_test_std)])
    wide = table.pivot(index="method", columns="domain", values="cell")[list(domains)]
    wide["Avg"] = [f"{summary[summary.method == m].acc_test_mean.mean() * 100:.2f}" for m in wide.index]
    print(f"{args.backbone}, test-set top-1 accuracy (%), mean ± std over {len(cfg['splits']['seeds'])} seeds")
    print(wide.to_string())
    print(summary[["method", "domain", "acc_all"]].to_string(index=False))


if __name__ == "__main__":
    main()
