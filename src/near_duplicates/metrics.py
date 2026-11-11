"""The shared per-query results schema, threshold calibration at a false-match rate, and recall with intervals.

Per-condition rates use Wilson intervals, since each condition queries every source image once. Rates pooled
over conditions use a bootstrap over source images, since each source is queried under every condition.

Every system writes one row per query, system and mode with RESULT_COLUMNS. Distractor queries have an
empty `true_id`; a query with no LSH candidates has an empty `predicted` and a score of -inf. Scores are
"higher is more similar" for every system, so one threshold rule serves all of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from near_duplicates.lsh import bit_sampling_curve, s_curve
from near_duplicates.phash import HASH_BITS

RESULT_COLUMNS = [
    "system", "mode", "kind", "strength", "query_id", "true_id", "predicted", "score",
    "candidates", "true_is_candidate", "true_similarity", "seconds",
]
GROUP = ["system", "mode", "kind", "strength"]
Z_95 = 1.959963984540054
BOOTSTRAP_RESAMPLES = 2000


def wilson_interval(successes: np.ndarray | int, n: np.ndarray | int, z: float = Z_95) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion; unlike the normal approximation it stays inside [0, 1]."""
    successes, n = np.asarray(successes, dtype=float), np.asarray(n, dtype=float)
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half_width = z / denominator * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return center - half_width, center + half_width


def source_resamples(source: np.ndarray, values: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0) -> np.ndarray:
    """Column totals of `values` (rows x columns) in bootstrap samples that resample whole source images.

    Each source image is queried under every transformation, so its rows are not independent: one source
    that resembles an indexed image produces false matches under many transformations at once. Drawing
    sources with replacement keeps that dependence, which a Wilson interval over the rows ignores.
    Returns shape (resamples, columns).
    """
    sources, inverse = np.unique(np.asarray(source), return_inverse=True)
    per_source = np.zeros((len(sources), values.shape[1]))
    np.add.at(per_source, inverse, values)
    counts = np.random.default_rng(seed).multinomial(len(sources), np.full(len(sources), 1 / len(sources)), size=resamples)
    return counts @ per_source


def percentile_interval(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return float(low), float(high)


def pooled_rates(results: pd.DataFrame, thresholds: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Per system and mode: recall at the threshold over all targets and the false-match rate over all
    distractors, with 95% intervals from resampling source images (`source_resamples`).

    Every condition has the same targets, so pooled recall equals the mean of the per-condition recalls.
    """
    df = results.merge(thresholds, on=["system", "mode"], validate="many_to_one")
    df = df.assign(target=is_target(df), accepted=df.score > df.threshold)
    rows = []
    for (system, mode), group in df.groupby(["system", "mode"], sort=False):
        targets, distractors = group[group.target], group[~group.target]
        hit = (targets.accepted & (targets.predicted == targets.true_id)).to_numpy(float)
        recall = source_resamples(targets.query_id, np.column_stack([hit, np.ones_like(hit)]), seed=seed)
        false = distractors.accepted.to_numpy(float)
        fmr = source_resamples(distractors.query_id, np.column_stack([false, np.ones_like(false)]), seed=seed)
        rows.append(dict(system=system, mode=mode, recall_at_threshold=hit.mean(),
                         **dict(zip(("recall_low", "recall_high"), percentile_interval(recall[:, 0] / recall[:, 1]))),
                         false_match_rate=false.mean(),
                         **dict(zip(("fmr_low", "fmr_high"), percentile_interval(fmr[:, 0] / fmr[:, 1])))))
    return pd.DataFrame(rows)


def threshold_at_fmr(distractor_scores: np.ndarray, fmr: float) -> float:
    """The lowest threshold t such that at most a fraction `fmr` of distractors score strictly above t.

    Ties are common (pHash scores are integers), so a plain quantile can overshoot the target rate.
    """
    return threshold_allowing(distractor_scores, int(np.floor(fmr * len(distractor_scores) + 1e-9)))


def threshold_allowing(distractor_scores: np.ndarray, allowed: int) -> float:
    """The lowest threshold t such that at most `allowed` distractors score strictly above t."""
    scores = np.sort(np.asarray(distractor_scores, dtype=float))
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


def candidate_probability(results: pd.DataFrame, bit_bands: int, bit_rows: int, minhash_bands: int, minhash_rows: int) -> np.ndarray:
    """Theoretical probability that each row's true match becomes an LSH candidate, from its true similarity."""
    pairs = results.true_similarity.to_numpy(float)
    probability = np.full(len(results), np.nan)
    phash = (results.system == "phash").to_numpy() & ~np.isnan(pairs)
    words = (results.system == "visual-words").to_numpy() & ~np.isnan(pairs)
    probability[phash] = bit_sampling_curve(np.rint(HASH_BITS * (1 - pairs[phash])).astype(int), bit_bands, bit_rows)
    probability[words] = s_curve(pairs[words], minhash_bands, minhash_rows)
    return probability


def lsh_loss(results: pd.DataFrame, bit_bands: int, bit_rows: int, minhash_bands: int, minhash_rows: int) -> pd.DataFrame:
    """Per system and condition: exhaustive and LSH recall@1, and the LSH recall the s-curve predicts (RQ2).

    LSH finds a target that exhaustive search finds exactly when the true match becomes a candidate, so the
    prediction is the mean over targets of [exhaustive correct] x P(candidate | true similarity). It ignores
    the rare query that LSH gets right because exhaustive search's wrong winner was not a candidate.
    """
    targets = results[is_target(results) & results.system.isin(["phash", "visual-words"])]
    keys = ["system", "kind", "strength", "query_id"]
    exhaustive = targets[targets["mode"] == "exhaustive"].set_index(keys)
    lsh = targets[targets["mode"] == "lsh"].set_index(keys)
    lsh = lsh.assign(
        exhaustive_correct=(exhaustive.predicted == exhaustive.true_id).reindex(lsh.index),
        lsh_correct=lsh.predicted == lsh.true_id,
        probability=candidate_probability(lsh.reset_index(), bit_bands, bit_rows, minhash_bands, minhash_rows),
    )
    lsh["predicted_lsh_correct"] = lsh.exhaustive_correct * lsh.probability
    return lsh.groupby(GROUP[:1] + GROUP[2:], sort=False).agg(
        queries=("lsh_correct", "size"),
        exhaustive_recall=("exhaustive_correct", "mean"),
        lsh_recall=("lsh_correct", "mean"),
        predicted_lsh_recall=("predicted_lsh_correct", "mean"),
        candidate_rate=("true_is_candidate", "mean"),
        predicted_candidate_rate=("probability", "mean"),
    ).reset_index()
