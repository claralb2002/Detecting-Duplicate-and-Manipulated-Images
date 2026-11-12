"""Analyses that need no new search, computed from a per-query results CSV written by `run`.

    uv run python -m near_duplicates.analyze results/queries_caltech101_n6209_k2048_p128_mb64r2_bb16r8_s0_val.csv

LSH loss (RQ2): measured LSH recall against the recall the s-curve predicts from each target's true similarity.
Combinations (RQ4): each system alone, the cascade, score fusion and the best possible combination, at the same
false-match rate. A test file is calibrated on the validation file of the same run and evaluated on the whole test
split, the protocol of the single systems; the DISC21 development queries (`_dev` files) are calibrated on
`dev_calibration` and evaluated on `dev_report`; any other file is calibrated on half of the source images and
evaluated on the other half.
Operating points (test files): every system and combination calibrated on validation at several target rates, with
the recall and false-match rate each realises on test, so methods are compared at equal realised rates.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from near_duplicates.cascade import SYSTEM_A, compare, compare_calibrated_on, operating_points, paired
from near_duplicates.metrics import lsh_loss
from near_duplicates.plots import plot_combinations, plot_operating_points
from near_duplicates.transforms import CONDITIONS

LSH_SETTINGS = re.compile(r"_mb(?P<minhash_bands>\d+)r(?P<minhash_rows>\d+)_bb(?P<bit_bands>\d+)r(?P<bit_rows>\d+)_")


def condition_order(frame: pd.DataFrame) -> pd.DataFrame:
    order = {condition: i for i, condition in enumerate(CONDITIONS)}
    return frame.sort_values("kind", key=lambda _: [order.get(c, -1) for c in zip(frame.kind, frame.strength)], kind="stable")


def by_condition(frame: pd.DataFrame, columns: str, values: str | list[str]) -> pd.DataFrame:
    """A pivot with one row per condition, in the order of `frame` (pivot_table itself reorders the rows)."""
    conditions = pd.MultiIndex.from_frame(frame[["kind", "strength"]].drop_duplicates())
    return frame.pivot_table(index=["kind", "strength"], columns=columns, values=values).reindex(conditions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queries", type=Path, help="a results/queries_<tag>.csv file")
    parser.add_argument("--target-fmr", type=float, default=0.01)
    args = parser.parse_args()
    settings = LSH_SETTINGS.search(args.queries.name)
    if settings is None:
        parser.error("the file name must carry the run tag with the LSH settings, as `run` writes it")
    tag = args.queries.stem.removeprefix("queries_")
    read = lambda path: pd.read_csv(path, keep_default_na=False, na_values={"true_similarity": [""]})
    results = read(args.queries)
    validation = None
    if tag.endswith("_test"):
        validation_file = args.queries.with_name(args.queries.name.removesuffix("_test.csv") + "_val.csv")
        if not validation_file.exists():
            parser.error(f"{validation_file} is missing: a test file is calibrated on the validation file of its run")
        validation = read(validation_file)
    out = args.queries.parent
    pd.set_option("display.width", 200)

    loss = condition_order(lsh_loss(results, **{k: int(v) for k, v in settings.groupdict().items()}))
    loss.to_csv(out / f"lsh_loss_{tag}.csv", index=False)
    pivot = by_condition(loss, "system", ["exhaustive_recall", "lsh_recall", "predicted_lsh_recall"])
    print(f"LSH loss: recall@1 exhaustive, measured LSH and predicted LSH ({tag})\n\n{pivot.round(2).to_string()}")
    totals = loss.groupby("system")[["exhaustive_recall", "lsh_recall", "predicted_lsh_recall", "candidate_rate",
                                     "predicted_candidate_rate"]].mean()
    print(f"\nMeans over conditions\n\n{totals.round(3).to_string()}")

    for system_b in [s for s in dict.fromkeys(results.system) if s != SYSTEM_A]:
        for mode in ("lsh", "exhaustive"):
            pairs = paired(results, mode, system_b)
            if validation is not None:
                per_condition, summary = compare_calibrated_on(paired(validation, mode, system_b), pairs, args.target_fmr,
                                                               system_b)
            else:
                calibrating = pairs.kind == "dev_calibration" if tag.endswith("_dev") else None
                per_condition, summary = compare(pairs, args.target_fmr, system_b, calibrating)
            per_condition = condition_order(per_condition)
            name = f"{tag}_{system_b}_{mode}"
            per_condition.to_csv(out / f"combinations_{name}.csv", index=False)
            summary.to_csv(out / f"combinations_summary_{name}.csv", index=False)
            recall = by_condition(per_condition, "method", "recall")
            evaluation = "calibrated on validation" if validation is not None else "held out"
            print(f"\nRecall at the {args.target_fmr:.0%} false-match rate, B = {system_b}, {mode} retrieval, {evaluation}\n\n"
                  f"{recall[summary.method].round(2).to_string()}\n\n{summary.round(4).to_string(index=False)}")
            plot_combinations(per_condition, out / f"combinations_{name}.png")
        if validation is not None:
            points = {mode: operating_points(paired(validation, mode, system_b), paired(results, mode, system_b), system_b)
                      for mode in ("exhaustive", "lsh")}
            pd.concat(points, names=["mode", None]).reset_index(level=0).to_csv(out / f"operating_points_{tag}_{system_b}.csv", index=False)
            plot_operating_points(points, out / f"operating_points_{tag}_{system_b}.png")


if __name__ == "__main__":
    main()
