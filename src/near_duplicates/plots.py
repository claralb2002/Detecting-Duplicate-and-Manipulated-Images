"""Figures from the shared results: recall against transformation strength, and LSH s-curves against measurement."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from near_duplicates.lsh import bit_sampling_curve, s_curve
from near_duplicates.phash import HASH_BITS

STYLES = {
    ("phash", "lsh"): ("C0", "-"), ("phash", "exhaustive"): ("C0", "--"),
    ("visual-words", "lsh"): ("C1", "-"), ("visual-words", "exhaustive"): ("C1", "--"),
    ("visual-words+ransac", "lsh"): ("C2", "-"), ("visual-words+ransac", "exhaustive"): ("C2", "--"),
}
UNGRADED = ("clean", "flip")


def plot_recall(summary: pd.DataFrame, path: Path) -> None:
    kinds = [k for k in dict.fromkeys(summary.kind) if k not in UNGRADED]
    fig, axes = plt.subplots(-(-len(kinds) // 4), 4, figsize=(15, 3.2 * -(-len(kinds) // 4)), squeeze=False)
    for ax, kind in zip(axes.flat, kinds):
        for (system, mode), (color, line) in STYLES.items():
            part = summary[(summary.kind == kind) & (summary.system == system) & (summary["mode"] == mode)].sort_values("strength")
            x = part.strength.astype(float)
            ax.plot(x, part.recall_at_1, color=color, linestyle=line, marker="o", markersize=3, label=f"{system} {mode}")
            ax.fill_between(x, part.recall_at_1_low, part.recall_at_1_high, color=color, alpha=0.12, linewidth=0)
        ax.set(title=kind, ylim=(0, 1.02), xlabel="strength", ylabel="recall@1")
    axes.flat[0].legend(fontsize=7)
    for ax in axes.flat[len(kinds) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_s_curves(results: pd.DataFrame, bit_bands: int, bit_rows: int, minhash_bands: int, minhash_rows: int, path: Path) -> None:
    """Observed P(true match is a candidate) per similarity bin against the theoretical candidate probability."""
    lsh_targets = results[(results["mode"] == "lsh") & (results.true_id != "")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    grid = np.linspace(0, 1, 201)
    distances = np.arange(HASH_BITS + 1)
    axes[0].plot(1 - distances / HASH_BITS, bit_sampling_curve(distances, bit_bands, bit_rows), "k-",
                 label=f"exact, b={bit_bands}, r={bit_rows}")
    axes[0].plot(grid, s_curve(grid, bit_bands, bit_rows), "k:", label="(1 - d/64)^r approximation")
    axes[1].plot(grid, s_curve(grid, minhash_bands, minhash_rows), "k-", label=f"theory, b={minhash_bands}, r={minhash_rows}")
    for ax, system, label in ((axes[0], "phash", "bit agreement 1 - d/64"), (axes[1], "visual-words", "Jaccard similarity")):
        part = lsh_targets[lsh_targets.system == system]
        bins = np.linspace(0, 1, 21)
        grouped = part.groupby(pd.cut(part.true_similarity, bins), observed=True).true_is_candidate.agg(["mean", "size"])
        centers = np.array([interval.mid for interval in grouped.index])
        ax.scatter(centers, grouped["mean"], s=5 + 40 * grouped["size"] / max(grouped["size"].max(), 1), color="C3",
                   label="observed (size = count)", zorder=3)
        ax.set(title=system, xlabel=label, ylabel="P(true match is a candidate)", ylim=(-0.02, 1.02))
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


COMBINATION_STYLES = {"cascade": ("C3", "-"), "fusion": ("C4", "--"), "either": ("k", ":")}


def plot_combinations(per_condition: pd.DataFrame, path: Path) -> None:
    """Recall at the common false-match rate against strength for each system alone and each combination."""
    kinds = [k for k in dict.fromkeys(per_condition.kind) if k not in UNGRADED]
    columns, rows = 3, -(-(len(kinds) + 1) // 3)
    fig, axes = plt.subplots(rows, columns, figsize=(13, 3.4 * rows), squeeze=False)
    for ax, kind in zip(axes.flat, kinds):
        styles = {m: COMBINATION_STYLES.get(m, (STYLES.get((m, "lsh"), ("C5", "-"))[0], "-")) for m in dict.fromkeys(per_condition.method)}
        for method, (color, line) in styles.items():
            part = per_condition[(per_condition.kind == kind) & (per_condition.method == method)].sort_values("strength")
            x = part.strength.astype(float)
            ax.plot(x, part.recall, color=color, linestyle=line, marker="o", markersize=3, label=method)
            if method != "either":
                ax.fill_between(x, part.recall_low, part.recall_high, color=color, alpha=0.1, linewidth=0)
        ax.set(title=kind, ylim=(0, 1.02), xlabel="strength", ylabel="recall at threshold")
    axes.flat[0].legend(fontsize=7)
    table_ax = axes.flat[len(kinds)]
    table_ax.axis("off")
    ungraded = per_condition[per_condition.kind.isin(UNGRADED)].pivot_table(index="kind", columns="method", values="recall")
    if not ungraded.empty:
        table_ax.table(cellText=ungraded.round(2).values, rowLabels=ungraded.index, colLabels=ungraded.columns, loc="center")
    for ax in axes.flat[len(kinds) + 1 :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_operating_points(points: dict[str, pd.DataFrame], path: Path) -> None:
    """Recall against the realised false-match rate per retrieval mode, one curve per system and combination;
    the points calibrated at a 1% target are circled."""
    fig, axes = plt.subplots(1, len(points), figsize=(5.5 * len(points), 4), squeeze=False)
    for ax, (mode, part) in zip(axes.flat, points.items()):
        for method in dict.fromkeys(part.method):
            color, line = COMBINATION_STYLES.get(method, (STYLES.get((method, "lsh"), ("C5", "-"))[0], "-"))
            curve = part[part.method == method].sort_values("target_fmr")
            ax.plot(curve.false_match_rate, curve.mean_recall, color=color, linestyle=line, marker=".", label=method)
            at_target = curve[np.isclose(curve.target_fmr, 0.01)]
            ax.scatter(at_target.false_match_rate, at_target.mean_recall, s=60, facecolors="none", edgecolors=color)
        ax.axvline(0.01, color="0.6", linewidth=0.8, linestyle=":")
        ax.set(xscale="log", title=f"{mode} retrieval", xlabel="realised false-match rate", ylabel="recall at threshold")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
