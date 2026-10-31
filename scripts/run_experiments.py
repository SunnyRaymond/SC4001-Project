"""Few-shot adapters A1, A2, A4 with leave-target-out selection (README 3.1, 3.3).

A1: L2-regularised multinomial logistic regression (sklearn lbfgs).
A2: linear probe initialised at the zero-shot prototypes, full-batch Adam.
A4: Tip-Adapter, training-free.

For target domain d and shot K, each grid config (configs/adapt.yaml) is scored
by simulating the same K-shot task on the other three domains with
splits.source_seeds; the config with the highest mean source accuracy is then
fitted once per seed on d's K-shot support and evaluated on d's test set.
d's test labels are used only in that final evaluation.

A source domain's simulated accuracy does not depend on the target, so every
(domain, K, seed, config) source run is done once and shared by all targets.

Writes results/<backbone>/:
  adapt_selection.csv   mean source accuracy of every config; the selected one flagged
  adapt_summary.csv     test accuracy of the selected config, per seed
  adapt_per_image.npz   "<M>/<slug>/K<k>/seed<s>/pred" int8 and ".../msp" float16, aligned
                        with the seed's test indices in splits/splits.npz
  adapt_probs.npz       full p_ad as float16 under "<M>/<slug>/K<k>/seed<s>" (not in git;
                        needed for gating and mixing)
"""

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import common

ADAPT_CONFIG = common.REPO_ROOT / "configs" / "adapt.yaml"


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def grid_configs(grid):
    """Grid points in a fixed order: product over keys in file order."""
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def edge_params(params, grid):
    """Parameters whose selected value is the smallest or largest of its grid."""
    return [k for k, v in params.items() if len(grid[k]) > 1 and v in (min(grid[k]), max(grid[k]))]


def fit_a1(Xs, ys, Xt, lam, max_iter):
    # sklearn minimises C * sum(loss) + ||W||^2 / 2, i.e. mean loss + lam/2 ||W||^2 with C = 1/(lam n).
    clf = LogisticRegression(C=1.0 / (lam * len(ys)), max_iter=max_iter)
    clf.fit(Xs, ys)
    return clf.decision_function(Xt)


def fit_a2(Xs, ys, Xt, T, scale, lr, steps, adam):
    b1, b2, eps = adam["beta1"], adam["beta2"], adam["eps"]
    Y = np.eye(T.shape[0], dtype=np.float32)[ys]
    W = T.astype(np.float32).copy()
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    for t in range(1, steps + 1):
        G = scale * (softmax(scale * Xs @ W.T) - Y).T @ Xs / len(ys)
        m = b1 * m + (1 - b1) * G
        v = b2 * v + (1 - b2) * G * G
        W -= lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    return scale * Xt @ W.T


def fit_a4(Xs, ys, Xt, T, scale, alpha, beta):
    L = np.eye(T.shape[0], dtype=np.float32)[ys]
    affinity = np.exp(-beta * (1 - Xt @ Xs.T))
    return scale * Xt @ T.T + alpha * affinity @ L


def fit_predict(method, params, mcfg, Xs, ys, Xt, T, scale):
    """Logits of one adapter on Xt after fitting on the support (Xs, ys)."""
    if method == "A1":
        return fit_a1(Xs, ys, Xt, params["lam"], mcfg["max_iter"])
    if method == "A2":
        return fit_a2(Xs, ys, Xt, T, scale, params["lr"], params["steps"], mcfg["adam"])
    if method == "A4":
        return fit_a4(Xs, ys, Xt, T, scale, params["alpha"], params["beta"])
    raise ValueError(method)


def k_shot(X, meta_labels, pool, test, k):
    """Support features/labels (first k of each class's pool) and test features/labels."""
    support = pool[:, :k].reshape(-1)
    ys = np.repeat(np.arange(pool.shape[0]), k)
    assert (meta_labels[support] == ys).all()
    return X[support], ys, X[test], meta_labels[test]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=common.DEFAULT_CONFIG)
    ap.add_argument("--adapt-config", default=ADAPT_CONFIG)
    ap.add_argument("--backbone", default="ViT-B-32-quickgelu")
    ap.add_argument("--methods", nargs="+", default=["A1", "A2", "A4"])
    ap.add_argument("--out", default=common.REPO_ROOT / "results")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    acfg = common.load_config(args.adapt_config)
    domains = cfg["dataset"]["domains"]
    shots, seeds, source_seeds = cfg["splits"]["shots"], cfg["splits"]["seeds"], cfg["splits"]["source_seeds"]
    feats = common.load_features(args.backbone)
    X = feats["image"].astype(np.float32)
    labels = feats["meta"]["label"].to_numpy()
    T = feats["text"][acfg["zero_shot_prototype"]].astype(np.float32)
    scale = feats["info"]["logit_scale"]
    splits = common.load_splits()

    sel_rows, sum_rows, per_image, probs_out = [], [], {}, {}
    for method in args.methods:
        mcfg = acfg["methods"][method]
        configs = grid_configs(mcfg["grid"])
        t0 = time.time()

        # Source simulations: acc[(slug, k)] -> [n_configs, n_source_seeds]
        acc = {}
        for slug, k in itertools.product(domains.values(), shots):
            a = np.empty((len(configs), len(source_seeds)))
            for j, seed in enumerate(source_seeds):
                Xs, ys, Xt, yt = k_shot(X, labels, *splits[(slug, seed)], k)
                for i, params in enumerate(configs):
                    a[i, j] = (fit_predict(method, params, mcfg, Xs, ys, Xt, T, scale).argmax(1) == yt).mean()
            acc[(slug, k)] = a
        print(f"{method}: {len(configs)} configs, source runs {time.time() - t0:.0f} s", flush=True)

        for (domain, slug), k in itertools.product(domains.items(), shots):
            src = np.mean([acc[(s, k)].mean(1) for s in domains.values() if s != slug], axis=0)
            best = int(np.argmax(src))  # first maximum = earlier in grid order
            for i, params in enumerate(configs):
                sel_rows.append({"method": method, "target": domain, "K": k, "config": json.dumps(params),
                                 "source_acc_mean": src[i], "selected": i == best})

            seed_acc = []
            for seed in seeds:
                Xs, ys, Xt, yt = k_shot(X, labels, *splits[(slug, seed)], k)
                p = softmax(fit_predict(method, configs[best], mcfg, Xs, ys, Xt, T, scale))
                key = f"{method}/{slug}/K{k}/seed{seed}"
                per_image[f"{key}/pred"] = p.argmax(1).astype(np.int8)
                per_image[f"{key}/msp"] = p.max(1).astype(np.float16)
                probs_out[key] = p.astype(np.float16)
                seed_acc.append((p.argmax(1) == yt).mean())
            sum_rows.append({
                "backbone": args.backbone, "method": method, "domain": domain, "K": k,
                "config": json.dumps(configs[best]), "at_grid_edge": ",".join(edge_params(configs[best], mcfg["grid"])),
                "acc_test_mean": np.mean(seed_acc), "acc_test_std": np.std(seed_acc, ddof=1),
                **{f"acc_test_seed{s}": a for s, a in zip(seeds, seed_acc)},
            })
        print(f"{method}: done in {time.time() - t0:.0f} s", flush=True)

    out = common.feature_dir(args.backbone, args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(sum_rows)
    pd.DataFrame(sel_rows).to_csv(out / "adapt_selection.csv", index=False, float_format="%.6f")
    summary.to_csv(out / "adapt_summary.csv", index=False, float_format="%.6f")
    np.savez_compressed(out / "adapt_per_image.npz", **per_image)
    np.savez_compressed(out / "adapt_probs.npz", **probs_out)

    # Test-set accuracy (%) averaged over seeds, B0/B1 for reference.
    zs = pd.read_csv(out / "zero_shot_summary.csv")
    table = summary.pivot_table(index=["method", "K"], columns="domain", values="acc_test_mean")[list(domains)]
    for m in ("B0", "B1"):
        table.loc[(m, 0), :] = zs[zs.method == m].set_index("domain").acc_test_mean[list(domains)].to_numpy()
    table["Avg"] = table.mean(axis=1)
    print(f"{args.backbone}, test-set top-1 accuracy (%), mean over {len(seeds)} seeds (K=0: zero-shot)")
    print((table.sort_index() * 100).round(2).to_string())
    print(summary[["method", "domain", "K", "config", "at_grid_edge"]].to_string(index=False))


if __name__ == "__main__":
    main()
