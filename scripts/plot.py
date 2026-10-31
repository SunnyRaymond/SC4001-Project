"""Report figures from the saved results (README section 7, figures 1-4).

Reads results/<backbone>/{zero_shot_summary, adapt_summary, flips, conf_bins,
gating_summary, gate_curves}.csv; writes figures/<backbone>/fig*.png and the
K=4 main table as results/<backbone>/main_table_k4.csv.
Colours: reference categorical slots 1-3 for A1/A2/A4, gray for zero-shot,
the blue/red diverging pair for help/harm.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import common

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
METHOD_COLOR = {"A1": "#2a78d6", "A2": "#eb6834", "A4": "#1baf7a"}
METHOD_NAME = {"A1": "A1 linear probe", "A2": "A2 ZS-init probe", "A4": "A4 Tip-Adapter"}
HELP, HARM = "#2a78d6", "#e34948"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "lines.linewidth": 2, "legend.frameon": False, "axes.axisbelow": True,
    })


def fig_accuracy(adapt, zs, domains, out):
    fig, axes = plt.subplots(1, len(domains), figsize=(10, 2.8))
    for ax, domain in zip(axes, domains):
        for m, color in METHOD_COLOR.items():
            d = adapt[(adapt.method == m) & (adapt.domain == domain)].sort_values("K")
            ax.errorbar(d.K, d.acc_test_mean * 100, yerr=d.acc_test_std * 100, color=color,
                        marker="o", markersize=5, capsize=2, label=METHOD_NAME[m])
        for m, ls in (("B0", "--"), ("B1", ":")):
            v = zs[(zs.method == m) & (zs.domain == domain)].acc_test_mean.iloc[0] * 100
            ax.axhline(v, color=INK2, ls=ls, lw=1.5, label=f"{m} zero-shot")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        ax.set_title(domain)
        ax.set_xlabel("shots per class K")
    axes[0].set_ylabel("test top-1 accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig1_accuracy_vs_k.png", dpi=200)
    plt.close(fig)


def fig_help_harm(flips, out):
    agg = flips.groupby(["method", "K"])[["help", "harm", "net"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=False)
    for ax, m in zip(axes, METHOD_COLOR):
        d = agg[agg.method == m]
        x = range(len(d))
        ax.bar(x, d.help * 100, width=0.6, color=HELP, label="help (B0 wrong, adapter right)")
        ax.bar(x, -d.harm * 100, width=0.6, color=HARM, label="harm (B0 right, adapter wrong)")
        ax.plot(x, d.net * 100, color=INK, marker="D", markersize=5, lw=0, label="net = help - harm")
        for xi, n in zip(x, d.net * 100):
            ax.annotate(f"{n:+.1f}", (xi, n), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK2)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(list(x), [f"K={k}" for k in d.K])
        ax.set_title(METHOD_NAME[m])
    axes[0].set_ylabel("% of test images")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig2_help_harm.png", dpi=200)
    plt.close(fig)


def fig_conf_bins(cbins, k, out):
    d = cbins[cbins.K == k].assign(h=lambda t: t.help * t.n, x=lambda t: t.harm * t.n, r=lambda t: t.acc_b0 * t.n)
    pooled = d.groupby(["method", "bin"])[["n", "h", "x", "r"]].sum().reset_index()
    pooled["net"] = (pooled.h - pooled.x) / pooled.n * 100
    pooled["cond_harm"] = pooled.x / pooled.r * 100
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, col, ylabel in ((axes[0], "net", "net gain in bin (pp)"),
                            (axes[1], "cond_harm", "conditional harm (% of B0-right)")):
        for m, color in METHOD_COLOR.items():
            p = pooled[pooled.method == m]
            ax.plot(p.bin + 1, p[col], color=color, marker="o", markersize=5, label=METHOD_NAME[m])
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("B0 confidence decile (1 = least confident)")
        ax.set_ylabel(ylabel)
    axes[0].axhline(0, color=INK2, lw=0.8)
    axes[0].axvspan(8.5, 10.5, color=GRID, alpha=0.6, lw=0)
    axes[0].text(9.5, axes[0].get_ylim()[1], "H1: top 20%", ha="center", va="top", fontsize=7, color=INK2)
    axes[1].legend(fontsize=7)
    fig.suptitle(f"K={k}, pooled over domains and seeds", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out / f"fig3_confidence_bins_k{k}.png", dpi=200)
    plt.close(fig)


def fig_gate_curves(curves, gating, k, out):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    for ax, m in zip(axes, METHOD_COLOR):
        c = curves[(curves.method == m) & (curves.K == k)].groupby("keep_frac")[["acc_G1", "acc_C2_expected"]].mean()
        g = gating[(gating.method == m) & (gating.K == k)]
        ax.plot(c.index * 100, c.acc_G1 * 100, color=METHOD_COLOR[m], label="G1 confidence gate")
        ax.plot(c.index * 100, c.acc_C2_expected * 100, color=INK2, ls="--", lw=1.5, label="C2 random gate")
        ax.axhline(g.acc_C1.mean() * 100, color=INK2, ls=":", lw=1.5, label="C1 global mix (selected w)")
        ax.plot(g.keep_frac.mean() * 100, g.acc_G1.mean() * 100, marker="o", markersize=8, color=METHOD_COLOR[m],
                markeredgecolor=SURFACE, markeredgewidth=2, lw=0, label="G1 at selected tau")
        ax.set_title(METHOD_NAME[m])
        ax.set_xlabel("images kept on B0 (%)")
    axes[0].set_ylabel("test accuracy (%)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4, fontsize=7)
    fig.suptitle(f"K={k}, mean over domains and seeds; 0% = ungated adapter, 100% = B0", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / f"fig4_gate_curves_k{k}.png", dpi=200)
    plt.close(fig)


def main_table(adapt, flips, gating, zs, domains, k, res):
    rows = []
    for domain in domains:
        rows.append({"method": "B0", "domain": domain,
                     "acc": zs[(zs.method == "B0") & (zs.domain == domain)].acc_test_mean.iloc[0], "harm": 0.0})
    for m in METHOD_COLOR:
        for domain in domains:
            f = flips[(flips.method == m) & (flips.domain == domain) & (flips.K == k)]
            g = gating[(gating.method == m) & (gating.domain == domain) & (gating.K == k)]
            rows.append({"method": m, "domain": domain, "acc": f.acc_ad.mean(), "harm": f.harm.mean()})
            for v in ("G1", "C1"):
                rows.append({"method": f"{m}+{v}", "domain": domain, "acc": g[f"acc_{v}"].mean(),
                             "harm": g[f"harm_{v}"].mean()})
    t = pd.DataFrame(rows)
    wide = t.pivot(index="method", columns="domain", values=["acc", "harm"]) * 100
    wide = wide.reindex(columns=[(a, d) for a in ("acc", "harm") for d in domains])
    wide[("acc", "Avg")] = wide["acc"].mean(axis=1)
    wide[("harm", "Avg")] = wide["harm"].mean(axis=1)
    order = ["B0"] + [f"{m}{s}" for m in METHOD_COLOR for s in ("", "+G1", "+C1")]
    wide = wide.loc[order].round(2)
    wide.to_csv(res / f"main_table_k{k}.csv")
    return wide


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=common.DEFAULT_CONFIG)
    ap.add_argument("--backbone", default="ViT-B-32-quickgelu")
    ap.add_argument("--results", default=common.REPO_ROOT / "results")
    ap.add_argument("--figures", default=common.REPO_ROOT / "figures")
    ap.add_argument("--k", type=int, default=4, help="K for figures 3, 4 and the main table")
    args = ap.parse_args()

    domains = list(common.load_config(args.config)["dataset"]["domains"])
    res = common.feature_dir(args.backbone, args.results)
    out = common.feature_dir(args.backbone, args.figures)
    out.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(res / f"{name}.csv")
    zs, adapt, flips = read("zero_shot_summary"), read("adapt_summary"), read("flips")
    cbins, gating, curves = read("conf_bins"), read("gating_summary"), read("gate_curves")

    style()
    fig_accuracy(adapt, zs, domains, out)
    fig_help_harm(flips, out)
    fig_conf_bins(cbins, args.k, out)
    fig_gate_curves(curves, gating, args.k, out)
    print(main_table(adapt, flips, gating, zs, domains, args.k, res).to_string())
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
