"""Figures from the shared results: recall against transformation strength, and LSH s-curves against measurement."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from near_duplicates.lsh import bit_sampling_curve, s_curve
from near_duplicates.phash import HASH_BITS

STYLES = {
    ("phash", "lsh"): ("C0", "-"), ("phash", "exhaustive"): ("C0", "--"),
    ("visual-words", "lsh"): ("C1", "-"), ("visual-words", "exhaustive"): ("C1", "--"),
    ("visual-words+ransac", "lsh"): ("C2", "-"), ("visual-words+ransac", "exhaustive"): ("C2", "--"),
}
UNGRADED = ("clean", "flip")
LABELS = {"phash": "pHash", "visual-words": "visual words", "visual-words+ransac": "visual words + RANSAC"}

# The report is A4 with 20 mm margins and two columns 7 mm apart. Figures are drawn at the width they are printed at,
# so the font sizes below are the printed sizes and nothing is scaled when the figure is included.
TEXT_WIDTH, COLUMN_WIDTH = 6.69, 3.21
REPORT_STYLE = {
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "lines.linewidth": 1.0, "lines.markersize": 2.5, "axes.linewidth": 0.6, "savefig.dpi": 300,
}

# Per edit: panel title, the unit of its strength, an offset that turns the strength into that unit, and whether a
# lower value is the stronger edit. Those axes are reversed, so every panel gets harder to the right.
EDIT_AXES = {
    "jpeg": ("JPEG", "quality", 0, True),
    "downscale": ("Downscale", "side kept", 0, True),
    "crop": ("Crop", "area kept", 0, True),
    "rotate": ("Rotation", "degrees", 0, False),
    "blur": ("Blur", "sigma (pixels)", 0, False),
    "brightness": ("Brightness", "gain", 1, False),
    "contrast": ("Contrast", "contrast removed", 0, False),
    "overlay": ("Overlay", "area covered", 0, False),
}


def plot_recall(summary: pd.DataFrame, path: Path) -> None:
    kinds = [k for k in dict.fromkeys(summary.kind) if k not in UNGRADED]
    rows = -(-len(kinds) // 4)
    with plt.rc_context(REPORT_STYLE):
        fig, axes = plt.subplots(rows, 4, figsize=(TEXT_WIDTH, 1.42 * rows + 0.3), sharey=True, squeeze=False)
        for ax, kind in zip(axes.flat, kinds):
            title, unit, offset, lower_is_harder = EDIT_AXES.get(kind, (kind.replace("+", " + ").replace("jpeg", "JPEG"), "level", 0, False))
            for (system, mode), (color, line) in STYLES.items():
                part = summary[(summary.kind == kind) & (summary.system == system) & (summary["mode"] == mode)].sort_values("strength")
                x = part.strength.astype(float) + offset
                ax.plot(x, part.recall_at_1, color=color, linestyle=line, marker="o")
                ax.fill_between(x, part.recall_at_1_low, part.recall_at_1_high, color=color, alpha=0.12, linewidth=0)
            ax.set(title=title, ylim=(0, 1.03), xlabel=unit)
            if kind not in EDIT_AXES:
                ax.set_xticks(sorted(summary[summary.kind == kind].strength.unique()))
            if lower_is_harder:
                ax.invert_xaxis()
        for ax in axes[:, 0]:
            ax.set_ylabel("recall@1")
        for ax in axes.flat[len(kinds) :]:
            ax.axis("off")
        systems = dict.fromkeys(system for system, _ in STYLES)
        handles = [Line2D([], [], color=STYLES[system, "lsh"][0], label=LABELS[system]) for system in systems]
        handles += [Line2D([], [], color="0.3", linestyle=line, label=label) for line, label in (("-", "LSH"), ("--", "exhaustive"))]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, handlelength=2.4)
        fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.6, w_pad=0.5)
        fig.savefig(path)
        plt.close(fig)


def plot_s_curves(results: pd.DataFrame, bit_bands: int, bit_rows: int, minhash_bands: int, minhash_rows: int, path: Path) -> None:
    """Observed P(true match is a candidate) per similarity bin against the theoretical candidate probability."""
    lsh_targets = results[(results["mode"] == "lsh") & (results.true_id != "")]
    grid = np.linspace(0, 1, 201)
    distances = np.arange(HASH_BITS + 1)
    with plt.rc_context(REPORT_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.5))
        axes[0].plot(1 - distances / HASH_BITS, bit_sampling_curve(distances, bit_bands, bit_rows), "k-",
                     label=f"exact, b={bit_bands}, r={bit_rows}")
        axes[0].plot(grid, s_curve(grid, bit_bands, bit_rows), "k:", label="(1 - d/64)^r approximation")
        axes[1].plot(grid, s_curve(grid, minhash_bands, minhash_rows), "k-", label=f"theory, b={minhash_bands}, r={minhash_rows}")
        panels = ((axes[0], "phash", "bit agreement 1 - d/64", "upper left"), (axes[1], "visual-words", "Jaccard similarity", "lower right"))
        for ax, system, label, legend_corner in panels:
            part = lsh_targets[lsh_targets.system == system]
            bins = np.linspace(0, 1, 21)
            grouped = part.groupby(pd.cut(part.true_similarity, bins), observed=True).true_is_candidate.agg(["mean", "size"])
            centers = np.array([interval.mid for interval in grouped.index])
            ax.scatter(centers, grouped["mean"], s=3 + 20 * grouped["size"] / max(grouped["size"].max(), 1), color="C3",
                       label="observed (size = count)", zorder=3)
            ax.set(title=LABELS[system], xlabel=label, ylabel="P(candidate)", ylim=(-0.03, 1.03))
            ax.legend(loc=legend_corner)
        fig.tight_layout(h_pad=0.8)
        fig.savefig(path)
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
    with plt.rc_context(REPORT_STYLE):
        fig, axes = plt.subplots(len(points), 1, figsize=(COLUMN_WIDTH, 1.5 * len(points) + 0.65), sharex=True, sharey=True, squeeze=False)
        for ax, (mode, part) in zip(axes.flat, points.items()):
            for method in dict.fromkeys(part.method):
                color, line = COMBINATION_STYLES.get(method, (STYLES.get((method, "lsh"), ("C5", "-"))[0], "-"))
                curve = part[part.method == method].sort_values("target_fmr")
                ax.plot(curve.false_match_rate, curve.mean_recall, color=color, linestyle=line, marker="o", label=LABELS.get(method, method))
                at_target = curve[np.isclose(curve.target_fmr, 0.01)]
                ax.scatter(at_target.false_match_rate, at_target.mean_recall, s=22, facecolors="none", edgecolors=color, linewidths=0.7)
            ax.axvline(0.01, color="0.6", linewidth=0.6, linestyle=":")
            ax.set(xscale="log", ylabel="recall at threshold")
            ax.set_title(f"{'LSH' if mode == 'lsh' else mode} retrieval", loc="left")
        axes.flat[-1].set_xlabel("realised false-match rate")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, columnspacing=1.2)
        fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.6)
        fig.savefig(path)
        plt.close(fig)


def main() -> None:
    """Redraw the report's three figures from the CSVs of one finished run, without searching again."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("tag", help="run tag, e.g. disc21_n50000_k8192_p256_mb113r2_bb17r16_t10_s0_test")
    parser.add_argument("out", type=Path, help="directory for the PDF figures")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--system-b", default="visual-words+ransac")
    args = parser.parse_args()
    bit_bands, bit_rows = map(int, re.search(r"_bb(\d+)r(\d+)", args.tag).groups())
    minhash_bands, minhash_rows = map(int, re.search(r"_mb(\d+)r(\d+)", args.tag).groups())
    args.out.mkdir(parents=True, exist_ok=True)

    plot_recall(pd.read_csv(args.results / f"summary_{args.tag}.csv"), args.out / "recall.pdf")
    points = pd.read_csv(args.results / f"operating_points_{args.tag}_{args.system_b}.csv")
    plot_operating_points({mode: points[points["mode"] == mode] for mode in dict.fromkeys(points["mode"])}, args.out / "operating_points.pdf")
    queries = pd.read_csv(args.results / f"queries_{args.tag}.csv", usecols=["system", "mode", "true_id", "true_is_candidate", "true_similarity"])
    plot_s_curves(queries.fillna({"true_id": ""}), bit_bands, bit_rows, minhash_bands, minhash_rows, args.out / "scurve.pdf")


if __name__ == "__main__":
    main()
