"""Run both systems on one split and evaluate them.

    uv run python -m near_duplicates.run caltech101 --split val                     # tune here
    uv run python -m near_duplicates.run caltech101 --split val --queries 20 --kinds jpeg crop   # smoke test
    uv run python -m near_duplicates.run caltech101 --split test                    # report here
    uv run python -m near_duplicates.run disc21 --split val --index-size 10000      # scaling point
    uv run python -m near_duplicates.run disc21 --split dev                         # DISC21 development queries

`val` and `test` transform each split's targets (indexed images) and distractors (images outside the index)
under every condition in `transforms.CONDITIONS`. Thresholds are calibrated on the full validation split
at the target false-match rate and reused unchanged for the test split, so test numbers are never tuned.
`dev` (DISC21 only) uses Meta's own development queries: thresholds are calibrated on `dev_calibration`
and scores are reported on `dev_report`, including the challenge's micro-average precision.

`--index-size` searches a seeded random subset of the index that always contains the split's targets;
the vocabulary is always built from the full index, so only the index size changes between scaling points.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import pandas as pd

from near_duplicates.datasets import DATA_DIR, REPO_ROOT, dev_ground_truth, load_split
from near_duplicates.features import cached_features, init_worker
from near_duplicates.metrics import RESULT_COLUMNS, calibrate, retrieval_scores, summarize
from near_duplicates.plots import plot_recall, plot_s_curves
from near_duplicates.systems import PhashSystem, VisualWordSystem, search
from near_duplicates.transforms import CONDITIONS, KINDS
from near_duplicates.vocabulary import cached_vocabulary

RESULTS_DIR = REPO_ROOT / "results"
DEV_PARTS = ("dev_calibration", "dev_report")


@dataclass
class QuerySet:
    kind: str
    strength: float
    transform: tuple[str, float]
    query_ids: list[str]
    true_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=("caltech101", "disc21"))
    parser.add_argument("--split", choices=("val", "test", "dev"), default="val")
    parser.add_argument("--queries", type=int, help="use only the first N targets and N distractors (smoke test)")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, help="run only these transformations")
    parser.add_argument("--index-size", type=int, help="search a random subset of the index of this size")
    parser.add_argument("--vocabulary", type=int, default=2048, help="visual words (k-means clusters)")
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--minhash-bands", type=int, default=64)
    parser.add_argument("--minhash-rows", type=int, default=2)
    parser.add_argument("--bit-bands", type=int, default=16)
    parser.add_argument("--bit-rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="seed for k-means, MinHash, bit sampling and index subsets")
    parser.add_argument("--target-fmr", type=float, default=0.01, help="false-match rate that sets each threshold")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    if args.split == "dev" and (args.dataset != "disc21" or args.kinds or args.index_size):
        parser.error("--split dev needs disc21 and takes neither --kinds nor --index-size")
    return args


def query_sets(args: argparse.Namespace) -> list[QuerySet]:
    if args.split == "dev":
        truth = dev_ground_truth()
        parts = {part: load_split(args.dataset, part)[: args.queries] for part in DEV_PARTS}
        return [QuerySet(part, 0, ("clean", 0), ids, [truth.get(q, "") for q in ids]) for part, ids in parts.items()]
    targets = load_split(args.dataset, f"{args.split}_targets")[: args.queries]
    distractors = load_split(args.dataset, f"{args.split}_distractors")[: args.queries]
    return [
        QuerySet(kind, strength, (kind, strength), targets + distractors, targets + [""] * len(distractors))
        for kind, strength in CONDITIONS if args.kinds is None or kind in args.kinds
    ]


def index_subset(index_ids: list[str], required: set[str], size: int | None, seed: int) -> np.ndarray:
    """Sorted positions of a random index subset of `size` images that contains every required image."""
    if size is None or size >= len(index_ids):
        return np.arange(len(index_ids))
    must = np.array([i for i, image_id in enumerate(index_ids) if image_id in required], dtype=np.int64)
    if size < len(must):
        raise ValueError(f"--index-size must be at least {len(must)} to hold every target")
    others = np.setdiff1d(np.arange(len(index_ids)), must)
    chosen = np.random.default_rng(seed).permutation(others)[: size - len(must)]
    return np.sort(np.concatenate([must, chosen]))


def run_tag(args: argparse.Namespace, index_size: int) -> str:
    return (f"{args.dataset}_n{index_size}_k{args.vocabulary}_p{args.num_perm}_mb{args.minhash_bands}r{args.minhash_rows}"
            f"_bb{args.bit_bands}r{args.bit_rows}_s{args.seed}")


def main() -> None:
    args = parse_args()
    full_index = load_split(args.dataset, "index")
    sets = query_sets(args)
    cache = DATA_DIR / "cache" / args.dataset
    positions = index_subset(full_index, {t for s in sets for t in s.true_ids if t}, args.index_size, args.seed)
    print(f"{len(positions):,} of {len(full_index):,} indexed images, {len(sets)} query sets "
          f"of {len(sets[0].query_ids):,} queries", file=sys.stderr)

    with Pool(args.workers, initializer=init_worker) as pool:
        full = cached_features(cache / f"index_n{len(full_index)}.npz",
                               [(args.dataset, i, "clean", 0) for i in full_index], pool, "index")
        centers = cached_vocabulary(cache / f"vocabulary_n{len(full_index)}_k{args.vocabulary}_s{args.seed}.npz",
                                    full.descriptors, args.vocabulary, args.seed)
        index = full.subset(positions)
        systems = [
            PhashSystem(index.hashes, args.bit_bands, args.bit_rows, seed=args.seed),
            VisualWordSystem(index.descriptors, centers, args.num_perm, args.minhash_bands, args.minhash_rows, seed=args.seed),
        ]
        indexed = set(index.ids)
        rows = []
        for s in sets:
            kind, strength = s.transform
            jobs = [(args.dataset, image_id, kind, strength) for image_id in s.query_ids]
            name = f"{args.split}_{s.kind}_{s.strength}_n{len(jobs)}.npz"
            queries = cached_features(cache / "queries" / name, jobs, pool, f"{s.kind} {s.strength}")
            true_ids = [t if t in indexed else "" for t in s.true_ids]
            rows += search(systems, index.ids, queries, true_ids, s.kind, s.strength)

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    costs = pd.DataFrame({"system": [s.name for s in systems], "index_size": len(index.ids),
                          "memory_bytes": [s.memory_bytes() for s in systems]})
    report(results, costs, args, run_tag(args, len(index.ids)))


def thresholds_for(results: pd.DataFrame, args: argparse.Namespace, tag: str) -> pd.DataFrame:
    if args.split == "dev":
        return calibrate(results[results.kind == "dev_calibration"], args.target_fmr)
    path = RESULTS_DIR / f"thresholds_{tag}_fmr{args.target_fmr}.csv"
    if args.split == "test":
        if not path.exists():
            sys.exit(f"{path} is missing: run the full validation split with the same settings first")
        return pd.read_csv(path)
    thresholds = calibrate(results, args.target_fmr)
    if args.queries is None and args.kinds is None:
        thresholds.to_csv(path, index=False)
    return thresholds


def report(results: pd.DataFrame, costs: pd.DataFrame, args: argparse.Namespace, run: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f"{run}_{args.split}" + (f"_q{args.queries}" if args.queries else "")
    results.to_csv(RESULTS_DIR / f"queries_{tag}.csv", index=False)
    thresholds = thresholds_for(results, args, run)
    reported = results[results.kind == "dev_report"] if args.split == "dev" else results

    order = {condition: i for i, condition in enumerate(CONDITIONS)}
    summary = summarize(reported, thresholds)
    summary = summary.sort_values("kind", key=lambda _: [order.get(c, -1) for c in zip(summary.kind, summary.strength)], kind="stable")
    summary.to_csv(RESULTS_DIR / f"summary_{tag}.csv", index=False)

    conditions = pd.MultiIndex.from_frame(summary[["kind", "strength"]].drop_duplicates())
    recall = summary.pivot_table(index=["kind", "strength"], columns=["system", "mode"], values="recall_at_1").reindex(conditions)
    print(f"\nRecall@1 per condition ({tag})\n\n{recall.round(2).to_string()}")
    source = {"val": "calibrated here", "test": "from validation", "dev": "from dev_calibration"}[args.split]
    print(f"\nAt the {args.target_fmr:.0%} false-match threshold ({source}):")
    overall = summary.groupby(["system", "mode"]).agg(
        mean_recall_at_threshold=("recall_at_threshold", "mean"),
        false_matches=("false_matches", "sum"),
        distractors=("distractors", "sum"),
    )
    overall["false_match_rate"] = overall.pop("false_matches") / overall.pop("distractors")
    if args.split == "dev":
        overall = overall.join(retrieval_scores(reported).set_index(["system", "mode"]))
    print(overall.round(3).to_string())

    per_mode = results.groupby(["system", "mode"]).agg(candidates=("candidates", "mean"), ms_per_query=("seconds", "mean"))
    per_mode["ms_per_query"] *= 1000
    costs = costs.merge(per_mode.reset_index(), on="system")
    costs["fraction_of_index"] = costs.candidates / costs.index_size
    costs.to_csv(RESULTS_DIR / f"costs_{tag}.csv", index=False)
    print(f"\nSearch cost per query and index memory\n\n{costs.round(4).to_string(index=False)}")

    if args.split != "dev":
        plot_recall(summary, RESULTS_DIR / f"recall_{tag}.png")
    plot_s_curves(results, args.bit_bands, args.bit_rows, args.minhash_bands, args.minhash_rows, RESULTS_DIR / f"scurve_{tag}.png")


if __name__ == "__main__":
    main()
