"""Combining the two systems (RQ4): a cost-aware cascade, score fusion, and the best possible combination.

Everything is computed from the shared results rows, so no search is repeated. System A (pHash) answers
first; System B (visual words, with or without spatial verification) is consulted only when A's top score
is not above A's cascade threshold.
The two cascade thresholds are calibrated jointly so that the cascade as a whole meets one false-match rate.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from near_duplicates.metrics import (
    is_target,
    percentile_interval,
    source_resamples,
    threshold_allowing,
    threshold_at_fmr,
    wilson_interval,
)

SYSTEM_A = "phash"
PAIR_KEYS = ["mode", "kind", "strength", "query_id", "true_id"]


def paired(results: pd.DataFrame, mode: str, system_b: str) -> pd.DataFrame:
    """One row per query with A's and B's top result side by side: predicted_a, score_a, predicted_b, score_b."""
    part = results[results["mode"] == mode]
    a = part[part.system == SYSTEM_A].set_index(PAIR_KEYS)[["predicted", "score"]]
    b = part[part.system == system_b].set_index(PAIR_KEYS)[["predicted", "score"]]
    if not (a.index.is_unique and b.index.is_unique and a.index.sort_values().equals(b.index.sort_values())):
        raise ValueError("both systems must answer the same queries, once each")
    return a.add_suffix("_a").join(b.add_suffix("_b"), validate="one_to_one").reset_index()


def calibration_half(query_ids: pd.Series) -> pd.Series:
    """Split queries by source image, so a source image and all its transformed versions fall in one half."""
    return query_ids.map(lambda q: zlib.crc32(q.encode()) % 2 == 0)


@dataclass(frozen=True)
class CascadeThresholds:
    a: float
    b: float


def cascade_answer(pairs: pd.DataFrame, t: CascadeThresholds) -> tuple[pd.Series, pd.Series]:
    """The cascade's answer per query ("" when it rejects), and whether A alone settled the query."""
    settled_by_a = pairs.score_a > t.a
    accepted_b = ~settled_by_a & (pairs.score_b > t.b)
    answer = pairs.predicted_a.where(settled_by_a, pairs.predicted_b.where(accepted_b, ""))
    return answer, settled_by_a


def calibrate_cascade(pairs: pd.DataFrame, fmr: float) -> CascadeThresholds:
    """The threshold pair with the highest target recall whose false matches on distractors stay within `fmr`.

    For each candidate threshold on A, the false matches A lets through use up part of the budget, and B's
    threshold is the loosest that keeps the remaining distractors, those A passed on, within what is left.
    Ties in recall go to the pair that settles more queries with A alone, since that saves SIFT extraction.
    """
    distractors, targets = pairs[~is_target(pairs)], pairs[is_target(pairs)]
    budget = int(np.floor(fmr * len(distractors) + 1e-9))
    best, best_key = None, None
    for t_a in np.unique(np.concatenate([pairs.score_a.to_numpy(float), [np.inf]])):
        passed_on = distractors.score_a <= t_a
        spent = int((~passed_on).sum())
        if spent > budget:
            continue
        t = CascadeThresholds(float(t_a), threshold_allowing(distractors.score_b[passed_on].to_numpy(), budget - spent))
        answer, settled_by_a = cascade_answer(targets, t)
        key = ((answer == targets.true_id).sum(), settled_by_a.sum())
        if best_key is None or key > best_key:
            best, best_key = t, key
    return best


@dataclass(frozen=True)
class Fusion:
    """Each score becomes its tail probability among calibration distractors; the more surprising system answers."""
    sorted_a: np.ndarray
    sorted_b: np.ndarray
    threshold: float

    @staticmethod
    def tail(sorted_scores: np.ndarray, scores: pd.Series) -> np.ndarray:
        """Fraction of calibration distractors scoring at least `scores`."""
        return 1 - np.searchsorted(sorted_scores, scores.to_numpy(float), side="left") / len(sorted_scores)

    @classmethod
    def unthresholded(cls, distractors: pd.DataFrame) -> Fusion:
        return cls(np.sort(distractors.score_a.to_numpy(float)), np.sort(distractors.score_b.to_numpy(float)), -np.inf)

    def fused(self, pairs: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
        tail_a, tail_b = self.tail(self.sorted_a, pairs.score_a), self.tail(self.sorted_b, pairs.score_b)
        answer = pairs.predicted_a.where(tail_a <= tail_b, pairs.predicted_b)
        return answer, -np.minimum(tail_a, tail_b)

    def answer(self, pairs: pd.DataFrame) -> pd.Series:
        answer, score = self.fused(pairs)
        return answer.where(score > self.threshold, "")


def calibrate_fusion(pairs: pd.DataFrame, fmr: float) -> Fusion:
    distractors = pairs[~is_target(pairs)]
    fusion = Fusion.unthresholded(distractors)
    _, score = fusion.fused(distractors)
    return Fusion(fusion.sorted_a, fusion.sorted_b, threshold_at_fmr(score, fmr))


def compare(pairs: pd.DataFrame, fmr: float, system_b: str, calibrating: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate every combination on the `calibrating` rows and evaluate it on the rest; by default on one half
    of the source images and the other half.

    Returns recall per method and condition, and per method the pooled false-match rate and the share of
    queries settled without System B. "either" is the best possible combination: a target counts as found
    if either system finds it above its own threshold; it is an upper bound and does not meet `fmr`.
    Pooled rates carry 95% intervals from resampling source images, and so does each method's recall gain
    over System B alone, which is paired: both recalls come from the same resampled sources.
    """
    calibrating = calibration_half(pairs.query_id) if calibrating is None else calibrating
    fit, held_out = pairs[calibrating], pairs[~calibrating]
    fit_distractors = fit[~is_target(fit)]
    t_a = threshold_at_fmr(fit_distractors.score_a.to_numpy(), fmr)
    t_b = threshold_at_fmr(fit_distractors.score_b.to_numpy(), fmr)
    cascade, fusion = calibrate_cascade(fit, fmr), calibrate_fusion(fit, fmr)

    answer_a = held_out.predicted_a.where(held_out.score_a > t_a, "")
    answer_b = held_out.predicted_b.where(held_out.score_b > t_b, "")
    answer_cascade, settled_by_a = cascade_answer(held_out, cascade)
    answers = {
        SYSTEM_A: answer_a, system_b: answer_b, "cascade": answer_cascade, "fusion": fusion.answer(held_out),
        "either": answer_a.where((answer_a == held_out.true_id) & is_target(held_out) | (answer_b == ""), answer_b),
    }
    settled = {SYSTEM_A: 1.0, system_b: 0.0, "cascade": settled_by_a.mean(), "fusion": 0.0, "either": 0.0}

    target = is_target(held_out)
    per_condition, found, false = [], {}, {}
    for method, answer in answers.items():
        found[method] = ((answer == held_out.true_id) & target)[target].to_numpy(float)
        false[method] = (answer[~target] != "").to_numpy(float)
        counts = held_out[target][["kind", "strength"]].assign(found=found[method] > 0)
        part = counts.groupby(["kind", "strength"], sort=False).found.agg(hits="sum", queries="size").reset_index()
        part["recall"] = part.hits / part.queries
        part["recall_low"], part["recall_high"] = wilson_interval(part.hits, part.queries)
        per_condition.append(part.assign(method=method))

    # Every condition holds the same sources, so the mean recall over conditions is the pooled recall.
    methods = list(answers)
    recall = source_resamples(held_out.query_id[target], np.column_stack([found[m] for m in methods]))
    recall /= target.sum()
    fmr = source_resamples(held_out.query_id[~target], np.column_stack([false[m] for m in methods]))
    fmr /= (~target).sum()
    overall = []
    for i, method in enumerate(methods):
        overall.append(dict(
            method=method, mean_recall=found[method].mean(), **dict(zip(("recall_low", "recall_high"), percentile_interval(recall[:, i]))),
            gain_over_b=found[method].mean() - found[system_b].mean(),
            **dict(zip(("gain_low", "gain_high"), percentile_interval(recall[:, i] - recall[:, methods.index(system_b)]))),
            false_match_rate=false[method].mean(), **dict(zip(("fmr_low", "fmr_high"), percentile_interval(fmr[:, i]))),
            settled_without_b=settled[method]))
    thresholds = pd.DataFrame([dict(method=SYSTEM_A, a=t_a, b=np.nan), dict(method=system_b, a=np.nan, b=t_b),
                               dict(method="cascade", a=cascade.a, b=cascade.b)])
    summary = pd.DataFrame(overall).merge(thresholds, on="method", how="left")
    return pd.concat(per_condition, ignore_index=True), summary


def compare_calibrated_on(calibration: pd.DataFrame, evaluation: pd.DataFrame, fmr: float,
                          system_b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`compare` with every threshold set on the `calibration` pairs and every rate measured on all `evaluation` pairs.

    For a test run calibrated on its validation run, this is the protocol of the single systems, which reuse
    their validation thresholds on test, so the combinations and the systems alone are compared like for like.
    """
    both = pd.concat([calibration, evaluation], ignore_index=True)
    return compare(both, fmr, system_b, pd.Series(np.arange(len(both)) < len(calibration), index=both.index))
