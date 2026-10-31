"""VRF-style sample-wise weighting of zero-shot and adapter (configs/vrf.yaml, README 3.6, H3).

For each adapter M, target d and shot K, M is refitted with the config selected
for d. The failure set V holds the support images that M gets right and B0
gets wrong; a test image's weight on M is w(x) = sigmoid(-(d(x) - a) / b), with
d(x) the Euclidean distance to its k-th nearest neighbour in V, and the
prediction is argmax of w p_ad + (1 - w) p_zs. a, b are selected
leave-target-out on the source domains. V0 is the same rule with V replaced by
a random support subset of the same size.

Writes results/<backbone>/vrf_summary.csv: one row per (method, domain, K, seed)
with the selected a, b, |V|, mean weight, and accuracy / help / harm of VRF and V0,
next to ungated, G1 and C1 from gating_summary.csv.
"""

import argparse
import itertools
import json

import numpy as np
import pandas as pd

import common
import run_experiments as rx

VRF_CONFIG = common.REPO_ROOT / "configs" / "vrf.yaml"


def kth_distance(Xt, V, k):
    """Euclidean distance of each row of Xt (unit norm) to its k-th nearest row of V."""
    d = np.sqrt(np.clip(2 - 2 * Xt @ V.T, 0, None))
    return np.partition(d, k - 1, axis=1)[:, k - 1]


def vrf_weight(dist, a, b):
    return 1 / (1 + np.exp((dist - a) / b))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=common.DEFAULT_CONFIG)
    ap.add_argument("--adapt-config", default=rx.ADAPT_CONFIG)
    ap.add_argument("--vrf-config", default=VRF_CONFIG)
    ap.add_argument("--backbone", default="ViT-B-32-quickgelu")
    ap.add_argument("--results", default=common.REPO_ROOT / "results")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    acfg = common.load_config(args.adapt_config)
    vcfg = common.load_config(args.vrf_config)
    configs = rx.grid_configs(vcfg["grid"])
    repeats = acfg["gating"]["C2_repeats"]
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
    saved = np.load(res / "adapt_probs.npz")
    gating = pd.read_csv(res / "gating_summary.csv")
    methods = sorted({k.split("/")[0] for k in saved.files})
    p_zs_all = rx.softmax(scale * X @ T.T)

    def run(method, params, slug, k, seed):
        """Adapter on one K-shot task: failure-set features, test p_ad, p_zs, labels."""
        pool, test = splits[(slug, seed)]
        support = pool[:, :k].reshape(-1)
        Xs, ys, Xt, yt = rx.k_shot(X, labels, pool, test, k)
        mcfg = acfg["methods"][method]
        logits = rx.fit_predict(method, params, mcfg, Xs, ys, np.concatenate([Xs, Xt]), T, scale)
        p = rx.softmax(logits)
        p_sup, p_ad = p[: len(ys)], p[len(ys):]
        fail = (p_sup.argmax(1) == ys) & (p_zs_all[support].argmax(1) != ys)
        return Xs, fail, Xt, p_ad, p_zs_all[test], yt

    def k_of(n):
        return max(1, int(np.floor(vcfg["p_percent"] / 100 * n)))

    def vrf_pred(dist, p_ad, p_zs, a, b):
        if dist is None:  # empty failure set
            return p_zs.argmax(1), np.zeros(len(p_zs))
        w = vrf_weight(dist, a, b)
        return (w[:, None] * p_ad + (1 - w[:, None]) * p_zs).argmax(1), w

    cache, rows = {}, []
    for m_idx, method in enumerate(methods):
        for (d_idx, (domain, slug)), k in itertools.product(enumerate(domains.items()), shots):
            sel = selection[(selection.method == method) & (selection.target == domain)
                            & (selection.K == k) & selection.selected]
            params = json.loads(sel.config.iloc[0])

            src = []
            for s, seed in [(s, seed) for s in domains.values() if s != slug for seed in source_seeds]:
                key = (method, json.dumps(params), s, k, seed)
                if key not in cache:
                    Xs, fail, Xt, p_ad, p_zs, yt = run(method, params, s, k, seed)
                    dist = kth_distance(Xt, Xs[fail], k_of(fail.sum())) if fail.any() else None
                    cache[key] = [(vrf_pred(dist, p_ad, p_zs, c["a"], c["b"])[0] == yt).mean() for c in configs]
                src.append(cache[key])
            best = configs[int(np.argmax(np.mean(src, axis=0)))]

            for seed in seeds:
                Xs, fail, Xt, p_ad, p_zs, yt = run(method, params, slug, k, seed)
                stored = saved[f"{method}/{slug}/K{k}/seed{seed}"].astype(np.float32)
                assert np.abs(stored - p_ad).max() < 2e-3, "refit differs from adapt_probs.npz"
                zs_ok = p_zs.argmax(1) == yt
                kk = k_of(fail.sum())
                dist = kth_distance(Xt, Xs[fail], kk) if fail.any() else None
                pred, w = vrf_pred(dist, p_ad, p_zs, best["a"], best["b"])
                ok = pred == yt

                rng = np.random.default_rng([seed, m_idx, d_idx, k, 1])
                v0 = []
                for _ in range(repeats):
                    rand = rng.choice(len(Xs), fail.sum(), replace=False)
                    d0 = kth_distance(Xt, Xs[rand], kk) if fail.any() else None
                    v0.append((vrf_pred(d0, p_ad, p_zs, best["a"], best["b"])[0] == yt).mean())

                rows.append({"method": method, "domain": domain, "K": k, "seed": seed,
                             "a": best["a"], "b": best["b"], "fail_set": int(fail.sum()), "mean_w": w.mean(),
                             "acc_VRF": ok.mean(), "help_VRF": (~zs_ok & ok).mean(), "harm_VRF": (zs_ok & ~ok).mean(),
                             "acc_V0": np.mean(v0)})
        print(f"{method}: done", flush=True)

    keys = ["method", "domain", "K", "seed"]
    cols = ["acc_B0", "acc_ungated", "harm_ungated", "acc_G1", "acc_C1", "harm_C1", "w"]
    summary = pd.DataFrame(rows).merge(gating[keys + cols], on=keys)
    summary.to_csv(res / "vrf_summary.csv", index=False, float_format="%.6f")

    agg = summary.groupby(["method", "K"])[["acc_ungated", "acc_C1", "acc_G1", "acc_VRF", "acc_V0",
                                            "harm_ungated", "harm_C1", "harm_VRF", "mean_w"]].mean()
    print(f"{args.backbone}: test accuracy / harm (%), mean over domains and seeds")
    print((agg * 100).round(2).join(summary.groupby(["method", "K"]).fail_set.mean().round(1)).to_string())
    print("\nPaired difference (pp), mean ± std over (domain, K, seed); [wins/ties/losses]")
    for method in methods:
        s = summary[summary.method == method]
        cells = []
        for a, b in [("VRF", "C1"), ("VRF", "V0"), ("VRF", "G1"), ("VRF", "ungated")]:
            diff = (s[f"acc_{a}"] - s[f"acc_{b}"]) * 100
            cells.append(f"{a}-{b}: {diff.mean():+.2f} ± {diff.std():.2f} "
                         f"[{(diff > 1e-9).sum()}/{(diff.abs() <= 1e-9).sum()}/{(diff < -1e-9).sum()}]")
        print(f"{method}  " + "   ".join(cells))
    sel = summary.drop_duplicates(["method", "domain", "K"])
    print("\nSelected a (count over (domain, K))")
    print(sel.groupby("method").a.value_counts().unstack(fill_value=0).to_string())
    print(sel.groupby("method").b.value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
