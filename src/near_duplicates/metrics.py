"""The shared per-query results schema, threshold calibration at a false-match rate, and recall with Wilson intervals.

Every system writes one row per query, system and mode with RESULT_COLUMNS. Distractor queries have an
empty `true_id`; a query with no LSH candidates has an empty `predicted` and a score of -inf. Scores are
"higher is more similar" for every system, so one threshold rule serves all of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RESULT_COLUMNS = [
    "system", "mode", "kind", "strength", "query_id", "true_id", "predicted", "score",
    "candidates", "true_is_candidate", "true_similarity", "seconds",
]
GROUP = ["system", "mode", "kind", "strength"]
Z_95 = 1.959963984540054


def wilson_interval(successes: np.ndarray | int, n: np.ndarray | int, z: float = Z_95) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion; unlike the normal approximation it stays inside [0, 1]."""
    successes, n = np.asarray(successes, dtype=float), np.asarray(n, dtype=float)
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half_width = z / denominator * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return center - half_width, center + half_width


def threshold_at_fmr(distractor_scores: np.ndarray, fmr: float) -> float:
    """The lowest threshold t such that at most a fraction `fmr` of distractors score strictly above t.

    Ties are common (pHash scores are integers), so a plain quantile can overshoot the target rate.
    """
    scores = np.sort(np.asarray(distractor_scores, dtype=float))
    allowed = int(np.floor(fmr * len(scores) + 1e-9))
    if allowed >= len(scores):
        return -np.inf
    return float(scores[len(scores) - allowed - 1])


def micro_average_precision(correct: np.ndarray, scores: np.ndarray, positives: int) -> tuple[float, float]:
    """Micro-average precision and recall at 90% precision, as defined by the DISC21 evaluation code.

    All predictions are pooled and ranked by decreasing score; tied predictions are ranked with incorrect
    ones first, the worst case, as `facebookresearch/isc2021` does. `positives` counts queries that have a
    true match, so a positive query without a correct prediction lowers recall.
    """
    correct, scores = np.asarray(correct, dtype=bool), np.asarray(scores, dtype=float)
    order = np.lexsort((correct, -scores))
    true_positives = np.cumsum(correct[order])
    precision = true_positives / np.arange(1, len(order) + 1)
    recall = true_positives / positives
    average_precision = float((np.diff(recall, prepend=0) * precision).sum())
    reachable = precision >= 0.9
    return average_precision, float(recall[reachable].max()) if reachable.any() else 0.0


def is_target(results: pd.DataFrame) -> pd.Series:
    return results.true_id != ""


def retrieval_scores(results: pd.DataFrame) -> pd.DataFrame:
    """Micro-average precision and recall at 90% precision per system and mode, over every query in `results`."""
    def score(group: pd.DataFrame) -> pd.Series:
        predicted = group[group.predicted != ""]
        average_precision, recall_at_p90 = micro_average_precision(
            predicted.predicted == predicted.true_id, predicted.score, int(is_target(group).sum()))
        return pd.Series({"micro_ap": average_precision, "recall_at_p90": recall_at_p90})
    return results.groupby(["system", "mode"])[results.columns.tolist()].apply(score).reset_index()


def calibrate(results: pd.DataFrame, fmr: float) -> pd.DataFrame:
    """One threshold per system and mode from the distractor queries of all conditions pooled."""
    distractors = results[~is_target(results)]
    return (
        distractors.groupby(["system", "mode"]).score
        .apply(lambda scores: threshold_at_fmr(scores.to_numpy(), fmr))
        .rename("threshold").reset_index()
    )


def summarize(results: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """Per system, mode and condition: recall@1, recall at the threshold, false-match rate, with 95% Wilson intervals.

    Recall at the threshold counts a target as found only if the top result is correct and scores above the
    threshold. The false-match rate is the share of distractors whose top result scores above it.
    """
    df = results.merge(thresholds, on=["system", "mode"], validate="many_to_one")
    correct, accepted = df.predicted == df.true_id, df.score > df.threshold
    df = df.assign(target=is_target(df), correct=correct, accepted=accepted, accepted_hit=correct & accepted)
    targets, distractors = df[df.target], df[~df.target]
    summary = targets.groupby(GROUP, sort=False).agg(
        queries=("correct", "size"),
        hits=("correct", "sum"),
        accepted_hits=("accepted_hit", "sum"),
        candidates=("candidates", "mean"),
        true_similarity=("true_similarity", "mean"),
    )
    false_matches = distractors.groupby(GROUP, sort=False).agg(distractors=("accepted", "size"), false_matches=("accepted", "sum"))
    summary = summary.join(false_matches).reset_index()
    for name, successes, n in (
        ("recall_at_1", summary.hits, summary.queries),
        ("recall_at_threshold", summary.accepted_hits, summary.queries),
        ("false_match_rate", summary.false_matches, summary.distractors),
    ):
        summary[name] = successes / n
        summary[f"{name}_low"], summary[f"{name}_high"] = wilson_interval(successes, n)
    return summary
