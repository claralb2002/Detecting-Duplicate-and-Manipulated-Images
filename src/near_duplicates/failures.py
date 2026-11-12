"""Failure analysis: why each system misses, from its representation, and a local gallery of misses.

    uv run python -m near_duplicates.failures results/queries_<tag>_val.csv

The mechanism table gives, per condition and over the targets, the quantity each representation breaks on:
the Hamming distance of the true pair for pHash; the SIFT keypoints left in the query and the Jaccard
similarity of the true pair for visual words; and the ratio-test matches and RANSAC inliers of the true pair
for spatial verification (unique ratio-test matches on a sample of targets, since they are recomputed here).

The gallery samples misses of each system at the first strength of each transformation where its recall@1
falls below 0.5, and shows query, wrong answer and true source side by side, with each image's Flickr
attribution. It is written under `data/failures/`, never `results/`, because DISC21 images may not be
redistributed; the report may show at most 100 of them, each with its attribution.
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from near_duplicates.datasets import DATA_DIR, DISC21_DIR, image_path, load_split
from near_duplicates.features import cache_dir, load_features, source_image
from near_duplicates.metrics import is_target
from near_duplicates.phash import HASH_BITS
from near_duplicates.spatial import verify_spatial
from near_duplicates.transforms import CONDITIONS, query_seed, transform

RUN_FILE = re.compile(r"queries_(?P<dataset>[a-z0-9]+)_.*_(?P<split>val|test)$")
SYSTEMS = ("phash", "visual-words", "visual-words+ransac")
ORIGINAL_STRENGTH = {(kind, float(strength)): strength for kind, strength in CONDITIONS}


def mechanisms(results: pd.DataFrame, query_keypoints: dict[tuple[str, float], np.ndarray],
               matches: dict[tuple[str, float], np.ndarray]) -> pd.DataFrame:
    """Per condition: medians over targets of what each representation keeps of the true pair, and recall@1."""
    exhaustive = results[(results["mode"] == "exhaustive") & is_target(results)]
    true = exhaustive.pivot_table(index=["kind", "strength"], columns="system", values="true_similarity", aggfunc="median")
    recall = (exhaustive.assign(correct=exhaustive.predicted == exhaustive.true_id)
              .pivot_table(index=["kind", "strength"], columns="system", values="correct", aggfunc="mean"))
    rows = []
    for kind, strength in CONDITIONS:
        key = (kind, float(strength))
        if key not in true.index:
            continue
        rows.append(dict(
            kind=kind, strength=strength,
            phash_distance=HASH_BITS * (1 - true.loc[key, "phash"]), phash_recall=recall.loc[key, "phash"],
            query_keypoints=float(np.median(query_keypoints[key])), true_jaccard=true.loc[key, "visual-words"],
            words_recall=recall.loc[key, "visual-words"], ratio_matches=float(np.median(matches[key])),
            inliers=true.loc[key, "visual-words+ransac"], ransac_recall=recall.loc[key, "visual-words+ransac"],
        ))
    return pd.DataFrame(rows)


def breaking_conditions(results: pd.DataFrame, system: str) -> list[tuple[str, float]]:
    """For each transformation, the first condition in grid order where the system's recall@1 drops below 0.5."""
    part = results[(results.system == system) & (results["mode"] == "exhaustive") & is_target(results)]
    recall = (part.predicted == part.true_id).groupby([part.kind, part.strength]).mean()
    first = {}
    for kind, strength in CONDITIONS:
        key = (kind, float(strength))
        if kind not in first and key in recall.index and recall[key] < 0.5:
            first[kind] = key
    return list(first.values())


def attributions(image_ids: set[str]) -> pd.DataFrame:
    table = pd.read_csv(DISC21_DIR / "disc21_yfcc_attributions.csv")
    return table[table.disc21_id.isin(image_ids)]


def gallery(results: pd.DataFrame, system: str, per_condition: int, dataset: str, seed: int, out: Path) -> set[str]:
    """Save sampled misses of `system` as a grid of query, wrong answer and true source; returns the images shown."""
    rng = np.random.default_rng(seed)
    part = results[(results.system == system) & (results["mode"] == "exhaustive") & is_target(results)]
    chosen = []
    for kind, strength in breaking_conditions(results, system):
        misses = part[(part.kind == kind) & (part.strength == strength) & (part.predicted != part.true_id)]
        chosen += [misses.iloc[i] for i in rng.choice(len(misses), size=min(per_condition, len(misses)), replace=False)]
    if not chosen:
        return set()
    fig, axes = plt.subplots(len(chosen), 3, figsize=(7.5, 2.6 * len(chosen)), squeeze=False)
    for row, miss in zip(axes, chosen):
        # The grid's own strength value, not the CSV's float, keeps the query's seed and so its exact edit.
        strength = ORIGINAL_STRENGTH[(miss.kind, miss.strength)]
        query = transform(source_image(image_path(dataset, miss.query_id)), miss.kind, strength,
                          np.random.default_rng(query_seed(miss.kind, strength, miss.query_id)))
        panels = ((query, f"{miss.kind} {miss.strength:g}"), (source_image(image_path(dataset, miss.predicted)),
                  f"answer {miss.predicted}, score {miss.score:.3g}"), (source_image(image_path(dataset, miss.true_id)),
                  f"source {miss.true_id}, {miss.true_similarity:.3g}"))
        for ax, (img, title) in zip(row, panels):
            ax.imshow(img)
            ax.set_title(title, fontsize=7)
            ax.axis("off")
    fig.suptitle(f"{system}: misses where recall@1 first drops below 0.5", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / len(chosen)))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{system}.png", dpi=110)
    plt.close(fig)
    return {m.query_id for m in chosen} | {m.predicted for m in chosen}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queries", type=Path, help="a full results/queries_<tag>_<val|test>.csv")
    parser.add_argument("--match-sample", type=int, default=100, help="targets per condition whose ratio matches are counted")
    parser.add_argument("--per-condition", type=int, default=2, help="misses shown per breaking condition")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if (match := RUN_FILE.match(args.queries.stem)) is None:
        parser.error("expected a full val or test file named by `run`")
    dataset, split = match["dataset"], match["split"]
    tag = args.queries.stem.removeprefix("queries_")
    results = pd.read_csv(args.queries, keep_default_na=False, na_values={"true_similarity": [""]})
    results["strength"] = results.strength.astype(float)

    cache = cache_dir(dataset)
    full_index = load_split(dataset, "index")
    index = load_features(cache / f"index_n{len(full_index)}.npz", full_index)
    position = {image_id: i for i, image_id in enumerate(full_index)}
    targets, distractors = load_split(dataset, f"{split}_targets"), load_split(dataset, f"{split}_distractors")
    sample = np.random.default_rng(args.seed).choice(len(targets), size=min(args.match_sample, len(targets)), replace=False)
    keypoints, matches = {}, {}
    with ThreadPoolExecutor(args.workers) as pool:
        for kind, strength in CONDITIONS:
            queries = load_features(cache / "queries" / f"{split}_{kind}_{strength}_n{len(targets) + len(distractors)}.npz",
                                    targets + distractors)
            if queries is None:
                continue
            key = (kind, float(strength))
            keypoints[key] = np.array([len(d) for d in queries.descriptors[: len(targets)]])
            matches[key] = np.fromiter(pool.map(
                lambda i: verify_spatial({"keypoints": queries.keypoints[i], "descriptors": queries.descriptors[i]},
                                         {"keypoints": index.keypoints[position[targets[i]]],
                                          "descriptors": index.descriptors[position[targets[i]]]}).matches,
                sample.tolist()), int)
    table = mechanisms(results, keypoints, matches)
    table.to_csv(args.queries.parent / f"mechanisms_{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))

    out = DATA_DIR / "failures" / tag
    shown = set()
    for system in SYSTEMS:
        shown |= gallery(results, system, args.per_condition, dataset, args.seed, out)
    if dataset == "disc21" and shown:
        attributions(shown).to_csv(out / "attributions.csv", index=False)
    print(f"\ngallery of misses ({len(shown)} images, local only): {out}")


if __name__ == "__main__":
    main()
