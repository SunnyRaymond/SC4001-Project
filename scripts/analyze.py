"""Help/harm decomposition and zero-shot-confidence bins (README 3.4, RQ1, RQ2, H1).

Every adapter prediction is compared with B0 on the same test image:
  help = B0 wrong, adapter right;  harm = B0 right, adapter wrong (negative flip);
  net = help - harm = acc_ad - acc_B0;
  cond_help = help / #B0-wrong;    cond_harm = harm / #B0-right.
Confidence c(x) is B0's max softmax probability. Bins are 10 equal-frequency
bins of c(x) within each (domain, seed) test set, so they do not depend on K or
the adapter. H1: on the top 20% (bins 8-9), net <= 0; rejected for a
(method, domain, K) if net > 0 on most seeds.

Writes results/<backbone>/:
  flips.csv       one row per (method, domain, K, seed)
  conf_bins.csv   one row per (method, domain, K, seed, bin)
  h1.csv          top-20% net gain per (method, domain, K), seeds with net > 0
"""

import argparse

import numpy as np
import pandas as pd

import common

N_BINS = 10
TOP_BINS = (8, 9)  # top 20% of c(x)


def flip_counts(b0_ok, ad_ok):
    help_, harm = (~b0_ok & ad_ok).sum(), (b0_ok & ~ad_ok).sum()
    n_right = b0_ok.sum()
    return {
        "n": len(b0_ok), "acc_b0": b0_ok.mean(), "acc_ad": ad_ok.mean(),
        "help": help_ / len(b0_ok), "harm": harm / len(b0_ok), "net": (help_ - harm) / len(b0_ok),
        "cond_help": help_ / max(len(b0_ok) - n_right, 1), "cond_harm": harm / max(n_right, 1),
    }


def conf_bins(conf):
    """Equal-frequency bin id 0..N_BINS-1 of each confidence (ties broken by order)."""
    ranks = np.argsort(np.argsort(conf, kind="stable"), kind="stable")
    return ranks * N_BINS // len(conf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=common.DEFAULT_CONFIG)
    ap.add_argument("--backbone", default="ViT-B-32-quickgelu")
    ap.add_argument("--results", default=common.REPO_ROOT / "results")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    domains = cfg["dataset"]["domains"]
    shots, seeds = cfg["splits"]["shots"], cfg["splits"]["seeds"]
    res = common.feature_dir(args.backbone, args.results)
    zs = pd.read_csv(res / "zero_shot_per_image.csv", index_col="index")
    adapt = np.load(res / "adapt_per_image.npz")
    methods = sorted({k.split("/")[0] for k in adapt.files})
    splits = common.load_splits()

    flip_rows, bin_rows = [], []
    for (domain, slug), seed in [(d, s) for d in domains.items() for s in seeds]:
        test = splits[(slug, seed)][1]
        y = zs["label"].to_numpy()[test]
        b0_ok = zs["B0_pred"].to_numpy()[test] == y
        conf = zs["B0_msp"].to_numpy()[test]
        bins = conf_bins(conf)
        for method, k in [(m, k) for m in methods for k in shots]:
            ad_ok = adapt[f"{method}/{slug}/K{k}/seed{seed}/pred"].astype(np.int64) == y
            key = {"method": method, "domain": domain, "K": k, "seed": seed}
            flip_rows.append({**key, **flip_counts(b0_ok, ad_ok)})
            for b in range(N_BINS):
                m = bins == b
                bin_rows.append({**key, "bin": b, "conf_lo": conf[m].min(), "conf_hi": conf[m].max(),
                                 **flip_counts(b0_ok[m], ad_ok[m])})
    flips, cbins = pd.DataFrame(flip_rows), pd.DataFrame(bin_rows)

    top = cbins[cbins["bin"].isin(TOP_BINS)].copy()
    top_net = (top.assign(nn=top.net * top.n).groupby(["method", "domain", "K", "seed"])
               .apply(lambda g: g.nn.sum() / g.n.sum(), include_groups=False).rename("top20_net").reset_index())
    h1 = top_net.groupby(["method", "domain", "K"]).agg(
        top20_net_mean=("top20_net", "mean"), top20_net_std=("top20_net", "std"),
        seeds_net_pos=("top20_net", lambda v: int((v > 0).sum())), seeds=("top20_net", "size")).reset_index()
    h1["h1_rejected"] = h1.seeds_net_pos > h1.seeds / 2

    flips.to_csv(res / "flips.csv", index=False, float_format="%.6f")
    cbins.to_csv(res / "conf_bins.csv", index=False, float_format="%.6f")
    h1.to_csv(res / "h1.csv", index=False, float_format="%.6f")

    pct = lambda d: (d * 100).round(2)
    agg = flips.groupby(["method", "K"])[["acc_ad", "help", "harm", "net", "cond_harm"]].mean()
    print(f"{args.backbone}: help/harm vs B0 (%), mean over domains and seeds")
    print(pct(agg).to_string())
    print("\nH1, top-20% net gain (%) mean over seeds; [seeds with net > 0 / 5]")
    wide = h1.assign(cell=[f"{m * 100:+.2f} [{p}]" for m, p in zip(h1.top20_net_mean, h1.seeds_net_pos)])
    print(wide.pivot(index=["method", "K"], columns="domain", values="cell")[list(domains)].to_string())
    print(f"\nH1 rejected in {int(h1.h1_rejected.sum())} of {len(h1)} (method, domain, K) cells")
    pooled = cbins.assign(h=cbins.help * cbins.n, x=cbins.harm * cbins.n, r=cbins.acc_b0 * cbins.n).groupby(
        ["method", "bin"])[["n", "h", "x", "r"]].sum()
    pooled = pd.DataFrame({"net": (pooled.h - pooled.x) / pooled.n, "cond_harm": pooled.x / pooled.r,
                           "b0_acc": pooled.r / pooled.n})
    print("\nPer-bin net gain, conditional harm and B0 accuracy (%), pooled over domains, K and seeds")
    print(pct(pooled.unstack("method")).to_string())


if __name__ == "__main__":
    main()
