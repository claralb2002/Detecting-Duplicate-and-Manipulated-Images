"""Run both systems on one split under every transformation and evaluate them.

    uv run python -m near_duplicates.run caltech101 --split val                       # tune here
    uv run python -m near_duplicates.run caltech101 --split val --queries 20 --kinds jpeg crop   # smoke test
    uv run python -m near_duplicates.run caltech101 --split test                      # report here

Each split's targets (indexed images) and distractors (images outside the index) are transformed under
every condition in `transforms.CONDITIONS`. Thresholds are calibrated on the full validation split at the
target false-match rate and reused unchanged for the test split, so test numbers are never tuned.
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

import pandas as pd

from near_duplicates.datasets import DATA_DIR, REPO_ROOT, load_split
from near_duplicates.features import cached_features, init_worker
from near_duplicates.metrics import RESULT_COLUMNS, calibrate, summarize
from near_duplicates.plots import plot_recall, plot_s_curves
from near_duplicates.systems import PhashSystem, VisualWordSystem, search
from near_duplicates.transforms import CONDITIONS, KINDS
from near_duplicates.vocabulary import cached_vocabulary

RESULTS_DIR = REPO_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=("caltech101", "disc21"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--queries", type=int, help="use only the first N targets and N distractors (smoke test)")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, help="run only these transformations")
    parser.add_argument("--vocabulary", type=int, default=2048, help="visual words (k-means clusters)")
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--minhash-bands", type=int, default=64)
    parser.add_argument("--minhash-rows", type=int, default=2)
    parser.add_argument("--bit-bands", type=int, default=16)
    parser.add_argument("--bit-rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="seed for k-means, MinHash and bit sampling")
    parser.add_argument("--target-fmr", type=float, default=0.01, help="false-match rate that sets each threshold")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    return parser.parse_args()


def run_tag(args: argparse.Namespace) -> str:
    return (f"{args.dataset}_k{args.vocabulary}_p{args.num_perm}_mb{args.minhash_bands}r{args.minhash_rows}"
            f"_bb{args.bit_bands}r{args.bit_rows}_s{args.seed}")


def thresholds_for(results: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    path = RESULTS_DIR / f"thresholds_{run_tag(args)}_fmr{args.target_fmr}.csv"
    if args.split == "test":
        if not path.exists():
            sys.exit(f"{path} is missing: run the full validation split with the same settings first")
        return pd.read_csv(path)
    thresholds = calibrate(results, args.target_fmr)
    if args.queries is None and args.kinds is None:
        thresholds.to_csv(path, index=False)
    return thresholds


def main() -> None:
    args = parse_args()
    index_ids = load_split(args.dataset, "index")
    targets = load_split(args.dataset, f"{args.split}_targets")[: args.queries]
    distractors = load_split(args.dataset, f"{args.split}_distractors")[: args.queries]
    conditions = [(kind, strength) for kind, strength in CONDITIONS if args.kinds is None or kind in args.kinds]
    cache = DATA_DIR / "cache" / args.dataset
    print(f"{len(index_ids):,} indexed, {len(targets)} targets and {len(distractors)} distractors "
          f"under {len(conditions)} conditions", file=sys.stderr)

    with Pool(args.workers, initializer=init_worker) as pool:
        index = cached_features(cache / f"index_n{len(index_ids)}.npz",
                                [(args.dataset, i, "clean", 0) for i in index_ids], pool, "index")
        centers = cached_vocabulary(cache / f"vocabulary_n{len(index_ids)}_k{args.vocabulary}_s{args.seed}.npz",
                                    index.descriptors, args.vocabulary, args.seed)
        systems = [
            PhashSystem(index.hashes, args.bit_bands, args.bit_rows, seed=args.seed),
            VisualWordSystem(index.descriptors, centers, args.num_perm, args.minhash_bands, args.minhash_rows, seed=args.seed),
        ]
        rows = []
        for kind, strength in conditions:
            query_ids = targets + distractors
            jobs = [(args.dataset, image_id, kind, strength) for image_id in query_ids]
            name = f"{args.split}_{kind}_{strength}_n{len(query_ids)}.npz"
            queries = cached_features(cache / "queries" / name, jobs, pool, f"{kind} {strength}")
            rows += search(systems, index_ids, queries, targets + [""] * len(distractors), kind, strength)

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    report(results, args)


def report(results: pd.DataFrame, args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f"{run_tag(args)}_{args.split}" + (f"_q{args.queries}" if args.queries else "")
    results.to_csv(RESULTS_DIR / f"queries_{tag}.csv", index=False)
    thresholds = thresholds_for(results, args)
    order = {condition: i for i, condition in enumerate(CONDITIONS)}
    summary = summarize(results, thresholds)
    summary = summary.sort_values("kind", key=lambda _: [order[c] for c in zip(summary.kind, summary.strength)], kind="stable")
    summary.to_csv(RESULTS_DIR / f"summary_{tag}.csv", index=False)

    conditions = pd.MultiIndex.from_frame(summary[["kind", "strength"]].drop_duplicates())
    recall = summary.pivot_table(index=["kind", "strength"], columns=["system", "mode"], values="recall_at_1").reindex(conditions)
    print(f"\nRecall@1 per condition ({tag})\n\n{recall.round(2).to_string()}")
    print(f"\nAt the {args.target_fmr:.0%} false-match threshold ({'from validation' if args.split == 'test' else 'calibrated here'}):")
    overall = summary.groupby(["system", "mode"]).agg(
        mean_recall_at_threshold=("recall_at_threshold", "mean"),
        false_match_rate=("false_matches", "sum"),
        distractors=("distractors", "sum"),
    )
    overall["false_match_rate"] /= overall.pop("distractors")
    print(overall.round(3).to_string())
    lsh = results[results["mode"] == "lsh"].groupby("system").agg(candidates=("candidates", "mean"), seconds=("lsh_seconds", "mean"))
    lsh["fraction_of_index"] = lsh.candidates / len(load_split(args.dataset, "index"))
    print(f"\nLSH candidates per query\n\n{lsh.round(4).to_string()}")

    plot_recall(summary, RESULTS_DIR / f"recall_{tag}.png")
    plot_s_curves(results, args.bit_bands, args.bit_rows, args.minhash_bands, args.minhash_rows, RESULTS_DIR / f"scurve_{tag}.png")


if __name__ == "__main__":
    main()
