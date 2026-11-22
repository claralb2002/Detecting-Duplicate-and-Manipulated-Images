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
from near_duplicates.transforms import CHAINS

# The three system colours stay distinguishable under every colour-vision deficiency, which matplotlib's default
# orange and green (C1, C2) do not: under protanopia they are 0.7 apart in OKLab, so visual words and RANSAC merge.
PHASH_COLOR, VISUAL_WORDS_COLOR, RANSAC_COLOR = "#2a78d6", "#eb6834", "#1baf7a"
STYLES = {
    ("phash", "lsh"): (PHASH_COLOR, "-"), ("phash", "exhaustive"): (PHASH_COLOR, "--"),
    ("visual-words", "lsh"): (VISUAL_WORDS_COLOR, "-"), ("visual-words", "exhaustive"): (VISUAL_WORDS_COLOR, "--"),
    ("visual-words+ransac", "lsh"): (RANSAC_COLOR, "-"), ("visual-words+ransac", "exhaustive"): (RANSAC_COLOR, "--"),
}
MARKERS = {"phash": "o", "visual-words": "s", "visual-words+ransac": "^"}
UNGRADED = ("clean", "flip")
LABELS = {"phash": "pHash", "visual-words": "visual words", "visual-words+ransac": "visual words + RANSAC"}

# The report is A4 with 20 mm margins and two columns 7 mm apart. Figures are drawn at the width they are printed at,
# so the font sizes below are the printed sizes and nothing is scaled when the figure is included.
TEXT_WIDTH, COLUMN_WIDTH = 6.69, 3.21
REPORT_STYLE = {
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 7, "lines.linewidth": 1.0, "lines.markersize": 2.5, "axes.linewidth": 0.6, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.linewidth": 0.3, "grid.color": "0.88",
}

# Per edit: its name, the unit of its strength, an offset that turns the strength into that unit, and whether a lower
# value is the stronger edit. Those axes are reversed, so every panel gets harder to the right.
EDIT_AXES = {
    "jpeg": ("JPEG", "quality", 0, True),
    "downscale": ("Downscale", "side kept", 0, True),
    "crop": ("Crop", "area kept", 0, True),
    "rotate": ("Rotation", "degrees", 0, False),
    "blur": ("Blur", "sigma in pixels", 0, False),
    "brightness": ("Brightness", "gain", 1, False),
    "contrast": ("Contrast", "share removed", 0, False),
    "overlay": ("Overlay", "area covered", 0, False),
}
# Every level of this chain contains the single edit, so the single edit alone is drawn as the chain's level 0.
CHAIN_BASES = {"flip+jpeg": "flip"}


def chain_title(kind: str) -> str:
    names = [EDIT_AXES[step][0] if step in EDIT_AXES else step.capitalize() for step in kind.split("+")]
    return " + ".join([names[0], *(name if name.isupper() else name.lower() for name in names[1:])])


def chain_ticks(kind: str) -> dict[int, str]:
    """Each level labelled with its steps' strengths, one per line in title order and in the units of their single-edit
    panels; flips have none."""
    return {
        level: "\n".join(f"{strength + EDIT_AXES[step][2]:g}" for step, strength in steps if step in EDIT_AXES)
        for level, steps in enumerate(CHAINS[kind], start=1)
    }


def stacked_column_figure() -> tuple[plt.Figure, np.ndarray]:
    """Two panels stacked in one text column. The margins are fixed rather than fitted to the content, so two such
    figures placed in neighbouring columns have their panels at the same height."""
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.6))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.1, top=0.84, hspace=0.62)
    return fig, axes


def stacked_column_legend(fig: plt.Figure, handles: list, columns: int) -> None:
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 1.0), ncol=columns, frameon=False, handlelength=2.4, columnspacing=1.4)


def plot_recall(summary: pd.DataFrame, path: Path) -> None:
    kinds = [k for k in dict.fromkeys(summary.kind) if k not in UNGRADED]
    rows = -(-len(kinds) // 4)
    with plt.rc_context(REPORT_STYLE):
        fig, axes = plt.subplots(rows, 4, figsize=(TEXT_WIDTH, 1.25 * rows + 0.55), sharey=True, squeeze=False, layout="constrained")
        for ax, kind in zip(axes.flat, kinds):
            panel = summary[summary.kind == kind]
            if kind in CHAINS:
                title, offset, lower_is_harder, ticks = chain_title(kind), 0, False, chain_ticks(kind)
                base = summary[summary.kind == CHAIN_BASES.get(kind)]
                if len(base):
                    panel = pd.concat([base.assign(strength=0), panel])
                    ticks[0] = "none"
            else:
                name, unit, offset, lower_is_harder = EDIT_AXES[kind]
                title, ticks = f"{name} ({unit})", None
            for (system, mode), (color, line) in STYLES.items():
                part = panel[(panel.system == system) & (panel["mode"] == mode)].sort_values("strength")
                x = part.strength.astype(float) + offset
                ax.plot(x, part.recall_at_1, color=color, linestyle=line, marker=MARKERS[system])
                ax.fill_between(x, part.recall_at_1_low, part.recall_at_1_high, color=color, alpha=0.12, linewidth=0)
            ax.set(title=title, ylim=(0, 1.04), yticks=(0, 0.5, 1))
            if ticks:
                ax.set_xticks(list(ticks), list(ticks.values()))
                ax.margins(x=0.1)
            if lower_is_harder:
                ax.invert_xaxis()
        for ax in axes[:, 0]:
            ax.set_ylabel("recall@1")
        for ax in axes.flat[len(kinds) :]:
            ax.axis("off")
        systems = dict.fromkeys(system for system, _ in STYLES)
        handles = [Line2D([], [], color=STYLES[system, "lsh"][0], marker=MARKERS[system], label=LABELS[system]) for system in systems]
        handles += [Line2D([], [], color="0.3", linestyle=line, label=label) for line, label in (("-", "LSH"), ("--", "exhaustive"))]
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, handlelength=2.4, columnspacing=1.8)
        fig.savefig(path)
        plt.close(fig)


def plot_s_curves(results: pd.DataFrame, bit_bands: int, bit_rows: int, minhash_bands: int, minhash_rows: int, path: Path) -> None:
    """Observed P(true match is a candidate) per similarity bin against the theoretical candidate probability."""
    lsh_targets = results[(results["mode"] == "lsh") & (results.true_id != "")]
    grid = np.linspace(0, 1, 201)
    distances = np.arange(HASH_BITS + 1)
    with plt.rc_context(REPORT_STYLE):
        fig, axes = stacked_column_figure()
        theory, = axes[0].plot(1 - distances / HASH_BITS, bit_sampling_curve(distances, bit_bands, bit_rows), "k-", label="s-curve")
        approximation, = axes[0].plot(grid, s_curve(grid, bit_bands, bit_rows), "k:", label="approximation (1 - d/64)^r")
        axes[1].plot(grid, s_curve(grid, minhash_bands, minhash_rows), "k-")
        panels = (
            (axes[0], "phash", f"pHash, bit sampling (b = {bit_bands}, r = {bit_rows})", "bit agreement 1 - d/64"),
            (axes[1], "visual-words", f"visual words, MinHash (b = {minhash_bands}, r = {minhash_rows})", "Jaccard similarity"),
        )
        for ax, system, title, label in panels:
            part = lsh_targets[lsh_targets.system == system]
            bins = np.linspace(0, 1, 21)
            grouped = part.groupby(pd.cut(part.true_similarity, bins), observed=True).true_is_candidate.agg(["mean", "size"])
            centers = np.array([interval.mid for interval in grouped.index])
            observed = ax.scatter(centers, grouped["mean"], s=3 + 20 * grouped["size"] / max(grouped["size"].max(), 1), color="C3",
                                  label="observed", zorder=3)
            ax.set(title=title, xlabel=label, ylabel="P(proposed)", ylim=(-0.04, 1.04), xlim=(-0.02, 1.02))
        stacked_column_legend(fig, [theory, observed, approximation], columns=2)
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
        fig, axes = stacked_column_figure()
        for ax, (mode, part) in zip(axes, points.items()):
            for method in dict.fromkeys(part.method):
                color, line = COMBINATION_STYLES.get(method, (STYLES.get((method, "lsh"), ("C5", "-"))[0], "-"))
                curve = part[part.method == method].sort_values("target_fmr")
                ax.plot(curve.false_match_rate, curve.mean_recall, color=color, linestyle=line, marker="o", label="RANSAC" if method == "visual-words+ransac" else LABELS.get(method, method))
                at_target = curve[np.isclose(curve.target_fmr, 0.01)]
                ax.scatter(at_target.false_match_rate, at_target.mean_recall, s=22, facecolors="none", edgecolors=color, linewidths=0.7)
            ax.axvline(0.01, color="0.5", linewidth=0.6, linestyle=":")
            ax.set(xscale="log", title=f"{'LSH' if mode == 'lsh' else mode.capitalize()} retrieval", xlabel="realised false-match rate",
                   ylabel="recall at threshold", ylim=(0.38, 0.97))
        stacked_column_legend(fig, axes[0].get_legend_handles_labels()[0], columns=3)
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
