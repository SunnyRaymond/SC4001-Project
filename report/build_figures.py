"""Reformat committed result CSVs as vector figures at the report's print width.

No training, hyperparameter selection or result changes. Run:
    python3 report/build_figures.py
Produces report/figures/*.pdf. Original figures/ PNGs remain available.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "ViT-B-32-quickgelu"
OUT = ROOT / "report" / "figures"
COLORS = {"A1": "#27649e", "A2": "#be541e", "A4": "#16735a"}
WIDTH = 6.30  # 160 mm text width; labels retain their intended printed point size.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.labelsize": 8, "axes.titlesize": 9,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.20, "axes.axisbelow": True,
                     "pdf.fonttype": 42, "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"report/figures/{name}.pdf")


def shots(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_xlabel("Support images per class, K")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    adapt = pd.read_csv(RES / "adapt_summary.csv")
    zs = pd.read_csv(RES / "zero_shot_summary.csv")
    flips = pd.read_csv(RES / "flips.csv")
    bins = pd.read_csv(RES / "conf_bins.csv")
    gate = pd.read_csv(RES / "gating_summary.csv")
    curves = pd.read_csv(RES / "gate_curves.csv")
    signal = pd.read_csv(RES / "vrf_signal.csv")

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.65))
    for ax, domain in zip(axes.flat, ["Art", "Clipart", "Product", "Real World"]):
        for method, color in COLORS.items():
            d = adapt[(adapt.method == method) & (adapt.domain == domain)].sort_values("K")
            ax.errorbar(d.K, d.acc_test_mean * 100, yerr=d.acc_test_std * 100,
                        color=color, marker="o", markersize=3, capsize=2, label=method)
        for method, style in [("B0", "--"), ("B1", ":")]:
            val = zs[(zs.method == method) & (zs.domain == domain)].acc_test_mean.iloc[0]
            ax.axhline(val * 100, color="#555555", linestyle=style, label=method)
        shots(ax)
        ax.set_title(domain)
        ax.set_ylabel("Accuracy (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92), pad=0.6)
    save(fig, "accuracy")

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.15))
    agg = flips.groupby(["method", "K"])[["help", "harm", "net"]].mean()
    for ax, method in zip(axes, COLORS):
        d = agg.loc[method].sort_index() * 100
        x = np.arange(4)
        ax.bar(x, d.help, color="#27649e", label="Help")
        ax.bar(x, -d.harm, color="#bf4a4a", label="Harm")
        ax.plot(x, d.net, "kD", markersize=3, label="Net change")
        ax.axhline(0, color="#555555", linewidth=0.7)
        ax.set_xticks(x, ["1", "2", "4", "8"])
        ax.set_title(method)
        ax.set_xlabel("Shots K")
    axes[0].set_ylabel("Test images (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1), pad=0.6)
    save(fig, "flips")

    d = bins[bins.K == 4].copy()
    d["help_n"], d["harm_n"], d["correct_n"] = d.help * d.n, d.harm * d.n, d.acc_b0 * d.n
    pooled = d.groupby(["method", "bin"])[["n", "help_n", "harm_n", "correct_n"]].sum()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.35))
    for method, color in COLORS.items():
        d = pooled.loc[method]
        axes[0].plot(d.index + 1, (d.help_n - d.harm_n) / d.n * 100,
                     color=color, marker="o", markersize=3, label=method)
        axes[1].plot(d.index + 1, d.harm_n / d.correct_n * 100,
                     color=color, marker="o", markersize=3, label=method)
    axes[0].axhline(0, color="#555555", linewidth=0.7)
    axes[0].axvspan(8.5, 10.5, color="#dddddd", alpha=0.5)
    axes[0].set_ylabel("Net gain within bin (pp)")
    axes[1].set_ylabel("Conditional harm (%)")
    for ax in axes:
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("B0 confidence decile (low to high)")
    axes[1].legend(frameon=False)
    fig.tight_layout(pad=0.6)
    save(fig, "confidence")

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.35))
    for ax, method in zip(axes, COLORS):
        c = curves[(curves.method == method) & (curves.K == 4)].groupby("keep_frac")[["acc_G1", "acc_C2_expected"]].mean()
        g = gate[(gate.method == method) & (gate.K == 4)]
        ax.plot(c.index * 100, c.acc_G1 * 100, color=COLORS[method], label="G1 confidence gate")
        ax.plot(c.index * 100, c.acc_C2_expected * 100, color="#555555", linestyle="--", label="C2 random gate")
        ax.axhline(g.acc_C1.mean() * 100, color="#111111", linestyle=":", label="C1 fixed mix")
        ax.scatter([g.keep_frac.mean() * 100], [g.acc_G1.mean() * 100], color=COLORS[method],
                   s=24, edgecolors="white", zorder=3, label="Source-selected G1")
        ax.set_title(method)
        ax.set_xlabel("Kept on B0 (%)")
        ax.set_xticks([0, 50, 100])
    axes[0].set_ylabel("Accuracy (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.22, 1, 1), pad=0.6)
    save(fig, "gates")

    fig, ax = plt.subplots(figsize=(4.9, 2.6))
    cols = [("auroc_msp", "#555555", "B0 uncertainty", "--"),
            ("auroc_vrf_dist", COLORS["A2"], "A2 failure-set distance", "-"),
            ("auroc_random_dist", "#777777", "A2 random-set distance", ":")]
    agg = signal[signal.method == "A2"].groupby("K")
    for col, color, label, linestyle in cols:
        d = agg[col].agg(["mean", "std"])
        ax.errorbar(d.index, d["mean"], yerr=d["std"], marker="o", markersize=3,
                    capsize=2, color=color, linestyle=linestyle, label=label)
    shots(ax)
    ax.set_ylabel("AUROC for B0 errors")
    ax.set_ylim(0.45, 0.97)
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout(pad=0.6)
    save(fig, "signal")


if __name__ == "__main__":
    main()
