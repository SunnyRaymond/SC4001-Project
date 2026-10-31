"""Diagnostic for H3: how well each per-image signal separates B0-wrong from B0-right test images.

Signals: B0's max softmax probability (low = likely wrong) and the VRF distance
to the failure set (low = near a support image B0 got wrong), plus the same
distance to a random support subset of equal size. Reported as AUROC for
"B0 is wrong" on every (method, domain, K, seed) test set; 0.5 = no signal.

Writes results/<backbone>/vrf_signal.csv.
"""

import argparse
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common
import run_experiments as rx
import vrf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=common.DEFAULT_CONFIG)
    ap.add_argument("--adapt-config", default=rx.ADAPT_CONFIG)
    ap.add_argument("--vrf-config", default=vrf.VRF_CONFIG)
    ap.add_argument("--backbone", default="ViT-B-32-quickgelu")
    ap.add_argument("--results", default=common.REPO_ROOT / "results")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    acfg = common.load_config(args.adapt_config)
    vcfg = common.load_config(args.vrf_config)
    domains = cfg["dataset"]["domains"]
    feats = common.load_features(args.backbone)
    X = feats["image"].astype(np.float32)
    labels = feats["meta"]["label"].to_numpy()
    T = feats["text"][acfg["zero_shot_prototype"]].astype(np.float32)
    p_zs_all = rx.softmax(feats["info"]["logit_scale"] * X @ T.T)
    splits = common.load_splits()
    res = common.feature_dir(args.backbone, args.results)
    probs = np.load(res / "adapt_probs.npz")
    selection = pd.read_csv(res / "adapt_selection.csv")

    rows = []
    for method, (domain, slug), k, seed in itertools.product(
            ("A1", "A2", "A4"), domains.items(), cfg["splits"]["shots"], cfg["splits"]["seeds"]):
        pool, test = splits[(slug, seed)]
        Xs, ys, Xt, yt = rx.k_shot(X, labels, pool, test, k)
        params = json.loads(selection[(selection.method == method) & (selection.target == domain)
                                      & (selection.K == k) & selection.selected].config.iloc[0])
        p_sup = rx.softmax(rx.fit_predict(method, params, acfg["methods"][method], Xs, ys, Xs, T,
                                          feats["info"]["logit_scale"]))
        fail = (p_sup.argmax(1) == ys) & (p_zs_all[pool[:, :k].reshape(-1)].argmax(1) != ys)
        zs_wrong = p_zs_all[test].argmax(1) != yt
        kk = max(1, int(np.floor(vcfg["p_percent"] / 100 * fail.sum())))
        rand = np.random.default_rng([seed, k]).choice(len(Xs), fail.sum(), replace=False)
        rows.append({"method": method, "domain": domain, "K": k, "seed": seed,
                     "auroc_msp": roc_auc_score(zs_wrong, -p_zs_all[test].max(1)),
                     "auroc_vrf_dist": roc_auc_score(zs_wrong, -vrf.kth_distance(Xt, Xs[fail], kk)),
                     "auroc_random_dist": roc_auc_score(zs_wrong, -vrf.kth_distance(Xt, Xs[rand], kk)),
                     # Harm needs B0 right and the adapter wrong; does distance flag where the adapter helps?
                     "auroc_vrf_dist_help": roc_auc_score(
                         zs_wrong & (probs[f"{method}/{slug}/K{k}/seed{seed}"].argmax(1) == yt),
                         -vrf.kth_distance(Xt, Xs[fail], kk))})
    out = pd.DataFrame(rows)
    out.to_csv(res / "vrf_signal.csv", index=False, float_format="%.6f")
    print("AUROC for 'B0 wrong' (help: 'B0 wrong and adapter right'), mean over domains and seeds")
    print(out.groupby(["method", "K"])[["auroc_msp", "auroc_vrf_dist", "auroc_random_dist",
                                        "auroc_vrf_dist_help"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
