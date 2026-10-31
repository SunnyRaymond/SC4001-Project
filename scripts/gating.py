"""Confidence gating and its controls on saved adapter outputs (README 3.3, 3.4, RQ3, H2).

With p_zs from B0 and c(x) = max p_zs:
  G1  hard gate: B0's prediction if c(x) >= tau, else the adapter's
  G2  soft gate: c(x) * p_zs + (1 - c(x)) * p_ad (no hyperparameter)
  C1  global mix: w * p_zs + (1 - w) * p_ad
  C2  random gate keeping B0 on the same number of images as G1 (mean of C2_repeats draws)
  C3  oracle: right if B0 or the adapter is right (upper bound, not a method)
tau and w are selected leave-target-out like the adapters: for target d, shot K
and adapter M, M is refitted with the config selected for d (adapt_selection.csv)
on each other domain and splits.source_seeds, and the tau / w with the highest
mean source accuracy is kept. Target test labels are used only to evaluate.

Writes results/<backbone>/:
  gating_summary.csv  one row per (method, domain, K, seed): selected tau / w, keep
                      fraction and accuracy, help, harm of every variant
  gate_curves.csv     G1 and the expected C2 accuracy at keep fractions 0, 0.05, ..., 1
                      (analysis only: drawn with test labels)
"""

import argparse
import itertools
import json

import numpy as np
import pandas as pd

import common
import run_experiments as rx

KEEP_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def gate_g1(zs_pred, ad_pred, conf, tau):
    return np.where(conf >= tau, zs_pred, ad_pred)


def mix(p_zs, p_ad, w):
    return (w * p_zs + (1 - w) * p_ad).argmax(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=common.DEFAULT_CONFIG)
    ap.add_argument("--adapt-config", default=rx.ADAPT_CONFIG)
    ap.add_argument("--backbone", default="ViT-B-32-quickgelu")
    ap.add_argument("--results", default=common.REPO_ROOT / "results")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    acfg = common.load_config(args.adapt_config)
    gcfg = acfg["gating"]
    taus, ws = gcfg["G1_tau"], gcfg["C1_w"]
    domains = cfg["dataset"]["domains"]
    shots, seeds, source_seeds = cfg["splits"]["shots"], cfg["splits"]["seeds"], cfg["splits"]["source_seeds"]
    feats = common.load_features(args.backbone)
    X = feats["image"].astype(np.float32)
    labels = feats["meta"]["label"].to_numpy()
    T = feats["text"][acfg["zero_shot_prototype"]].astype(np.float32)
    scale = feats["info"]["logit_scale"]
    splits = common.load_splits()
    res = common.feature_dir(args.backbone, args.results)
    selection = pd.read_csv(res / "adapt_selection.csv")
    probs = np.load(res / "adapt_probs.npz")
    methods = sorted({k.split("/")[0] for k in probs.files})

    p_zs_all = rx.softmax(scale * X @ T.T)
    zs_pred_all, conf_all = p_zs_all.argmax(1), p_zs_all.max(1)

    # Source refits, shared by every target that selected the same adapter config.
    cache = {}

    def source_scores(method, params, slug, k, seed):
        key = (method, json.dumps(params), slug, k, seed)
        if key not in cache:
            Xs, ys, Xt, yt = rx.k_shot(X, labels, *splits[(slug, seed)], k)
            test = splits[(slug, seed)][1]
            p_ad = rx.softmax(rx.fit_predict(method, params, acfg["methods"][method], Xs, ys, Xt, T, scale))
            ad_pred = p_ad.argmax(1)
            cache[key] = (
                [(gate_g1(zs_pred_all[test], ad_pred, conf_all[test], t) == yt).mean() for t in taus],
                [(mix(p_zs_all[test], p_ad, w) == yt).mean() for w in ws],
            )
        return cache[key]

    rows, curves = [], []
    for m_idx, method in enumerate(methods):
        for (d_idx, (domain, slug)), k in itertools.product(enumerate(domains.items()), shots):
            sel = selection[(selection.method == method) & (selection.target == domain)
                            & (selection.K == k) & selection.selected]
            params = json.loads(sel.config.iloc[0])
            scores = [source_scores(method, params, s, k, seed)
                      for s in domains.values() if s != slug for seed in source_seeds]
            tau = taus[int(np.argmax(np.mean([g for g, _ in scores], axis=0)))]
            w = ws[int(np.argmax(np.mean([c for _, c in scores], axis=0)))]

            for seed in seeds:
                test = splits[(slug, seed)][1]
                y, p_zs, conf = labels[test], p_zs_all[test], conf_all[test]
                zs_pred = zs_pred_all[test]
                p_ad = probs[f"{method}/{slug}/K{k}/seed{seed}"].astype(np.float32)
                ad_pred = p_ad.argmax(1)
                zs_ok, ad_ok = zs_pred == y, ad_pred == y

                keep = conf >= tau
                rng = np.random.default_rng([seed, m_idx, d_idx, k])
                c2 = []
                for _ in range(gcfg["C2_repeats"]):
                    rand_keep = np.zeros(len(y), bool)
                    rand_keep[rng.choice(len(y), keep.sum(), replace=False)] = True
                    c2.append(np.where(rand_keep, zs_ok, ad_ok).mean())

                preds = {
                    "ungated": ad_pred,
                    "G1": gate_g1(zs_pred, ad_pred, conf, tau),
                    "G2": (conf[:, None] * p_zs + (1 - conf[:, None]) * p_ad).argmax(1),
                    "C1": mix(p_zs, p_ad, w),
                }
                row = {"method": method, "domain": domain, "K": k, "seed": seed, "config": json.dumps(params),
                       "tau": tau, "w": w, "keep_frac": keep.mean(), "acc_B0": zs_ok.mean()}
                for name, pred in preds.items():
                    ok = pred == y
                    row[f"acc_{name}"] = ok.mean()
                    row[f"help_{name}"] = (~zs_ok & ok).mean()
                    row[f"harm_{name}"] = (zs_ok & ~ok).mean()
                row["acc_C2"] = np.mean(c2)
                row["acc_C3"] = (zs_ok | ad_ok).mean()
                rows.append(row)

                # Keep the q most confident images on B0; random keep of q has expectation q*acc_zs + (1-q)*acc_ad.
                order = np.argsort(-conf, kind="stable")
                for q in KEEP_GRID:
                    kept = np.zeros(len(y), bool)
                    kept[order[: int(round(q * len(y)))]] = True
                    curves.append({"method": method, "domain": domain, "K": k, "seed": seed, "keep_frac": q,
                                   "acc_G1": np.where(kept, zs_ok, ad_ok).mean(),
                                   "acc_C2_expected": q * zs_ok.mean() + (1 - q) * ad_ok.mean()})
        print(f"{method}: done", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(res / "gating_summary.csv", index=False, float_format="%.6f")
    pd.DataFrame(curves).to_csv(res / "gate_curves.csv", index=False, float_format="%.6f")

    variants = ["B0", "ungated", "G1", "G2", "C1", "C2", "C3"]
    agg = summary.groupby(["method", "K"])[[f"acc_{v}" for v in variants] + ["keep_frac", "harm_ungated", "harm_G1"]].mean()
    print(f"{args.backbone}: test accuracy (%), mean over domains and seeds")
    print((agg * 100).round(2).to_string())

    # H2: paired differences over seeds; win/tie/loss counted over (domain, K, seed).
    print("\nPaired difference (pp), mean ± std over (domain, K, seed); [wins/ties/losses]")
    for method in methods:
        s = summary[summary.method == method]
        cells = []
        for a, b in [("G1", "ungated"), ("G1", "C2"), ("G1", "C1"), ("G2", "C1")]:
            diff = (s[f"acc_{a}"] - s[f"acc_{b}"]) * 100
            cells.append(f"{a}-{b}: {diff.mean():+.2f} ± {diff.std():.2f} "
                         f"[{(diff > 1e-9).sum()}/{(diff.abs() <= 1e-9).sum()}/{(diff < -1e-9).sum()}]")
        print(f"{method}  " + "   ".join(cells))
    print("\nSelected tau and w (count over (domain, K))")
    sel = summary.drop_duplicates(["method", "domain", "K"])
    print(sel.groupby("method").tau.value_counts().unstack(fill_value=0).to_string())
    print(sel.groupby("method").w.value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
