from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from near_duplicates.cascade import (
    CascadeThresholds, calibrate_cascade, calibrate_fusion, calibration_half, cascade_answer, compare,
    compare_calibrated_on, paired,
)
from near_duplicates.metrics import RESULT_COLUMNS


def synthetic_pairs(n_targets: int, n_distractors: int, seed: int) -> pd.DataFrame:
    """Targets each system finds with some probability, integer A scores and continuous B scores, as in real runs."""
    rng = np.random.default_rng(seed)
    n = n_targets + n_distractors
    target = np.arange(n) < n_targets
    true_id = np.where(target, [f"r{i}" for i in range(n)], "")
    a_right, b_right = target & (rng.random(n) < 0.6), target & (rng.random(n) < 0.6)
    return pd.DataFrame({
        "mode": "lsh", "kind": "crop", "strength": 0.5, "query_id": [f"q{i}" for i in range(n)], "true_id": true_id,
        "predicted_a": np.where(a_right, true_id, "wrong"),
        "score_a": -rng.integers(0, 32, n) + np.where(a_right, 20, 0).astype(float),
        "predicted_b": np.where(b_right, true_id, "wrong"),
        "score_b": rng.random(n) * 0.3 + np.where(b_right, 0.25, 0),
    })


def brute_force_best_recall(pairs: pd.DataFrame, fmr: float) -> int:
    distractors, targets = pairs[pairs.true_id == ""], pairs[pairs.true_id != ""]
    budget = int(np.floor(fmr * len(distractors) + 1e-9))
    best = 0
    a_values = np.concatenate([np.unique(pairs.score_a), [np.inf]])
    b_values = np.concatenate([np.unique(pairs.score_b), [-np.inf, np.inf]])
    for t_a, t_b in itertools.product(a_values, b_values):
        t = CascadeThresholds(t_a, t_b)
        if (cascade_answer(distractors, t)[0] != "").sum() <= budget:
            best = max(best, int((cascade_answer(targets, t)[0] == targets.true_id).sum()))
    return best


@pytest.mark.parametrize("seed", range(5))
def test_joint_calibration_meets_the_budget_and_matches_brute_force(seed):
    pairs = synthetic_pairs(60, 200, seed)
    t = calibrate_cascade(pairs, fmr=0.05)
    distractors, targets = pairs[pairs.true_id == ""], pairs[pairs.true_id != ""]
    assert (cascade_answer(distractors, t)[0] != "").sum() <= 10
    assert (cascade_answer(targets, t)[0] == targets.true_id).sum() == brute_force_best_recall(pairs, 0.05)


def test_cascade_consults_b_only_when_a_is_not_confident():
    pairs = pd.DataFrame({"predicted_a": ["x", "y", "z"], "score_a": [-2.0, -20.0, -20.0],
                          "predicted_b": ["u", "v", "w"], "score_b": [0.9, 0.9, 0.1]})
    answer, settled_by_a = cascade_answer(pairs, CascadeThresholds(a=-10, b=0.5))
    assert answer.tolist() == ["x", "v", ""] and settled_by_a.tolist() == [True, False, False]


def test_fusion_answers_with_the_more_surprising_system_and_meets_the_rate():
    pairs = synthetic_pairs(100, 1000, seed=7)
    fusion = calibrate_fusion(pairs, fmr=0.02)
    distractors = pairs[pairs.true_id == ""]
    assert (fusion.answer(distractors) != "").mean() <= 0.02
    # A score above every distractor of its system has tail 0, so that system answers.
    probe = pd.DataFrame({"predicted_a": ["a", "a"], "score_a": [100.0, -40.0], "predicted_b": ["b", "b"], "score_b": [0.0, 10.0]})
    assert fusion.fused(probe)[0].tolist() == ["a", "b"]


def results_rows(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, suffix in (("phash", "a"), ("visual-words", "b")):
        part = pairs.rename(columns={f"predicted_{suffix}": "predicted", f"score_{suffix}": "score"})
        rows.append(part.assign(system=system, candidates=1, true_is_candidate=True, true_similarity=0.5, seconds=0.0))
    return pd.concat(rows)[RESULT_COLUMNS]


def test_paired_round_trips_and_rejects_unmatched_queries():
    pairs = synthetic_pairs(5, 5, seed=0)
    results = results_rows(pairs)
    back = paired(results, "lsh", "visual-words").set_index("query_id").loc[pairs.query_id]
    assert (back.score_a.to_numpy() == pairs.score_a.to_numpy()).all()
    assert (back.predicted_b.to_numpy() == pairs.predicted_b.to_numpy()).all()
    with pytest.raises(ValueError):
        paired(results.iloc[1:], "lsh", "visual-words")


def test_calibration_half_keeps_every_version_of_a_source_image_together():
    ids = pd.Series([f"img{i}" for i in range(200)] * 3)
    half = calibration_half(ids)
    assert 70 < half[:200].sum() < 130
    assert (half.groupby(ids).nunique() == 1).all()


def test_either_counts_a_false_match_from_either_system():
    pairs = pd.DataFrame({
        "mode": "lsh", "kind": "crop", "strength": 0.5, "true_id": ["", "t"] * 200,
        "query_id": [f"q{i}" for i in range(400)],
        "predicted_a": "x", "score_a": -30.0, "predicted_b": "y", "score_b": 0.0,
    })
    # Distractors accepted only by B in the held-out half: either must count them as false matches.
    held_out = ~calibration_half(pairs.query_id)
    pairs.loc[held_out & (pairs.true_id == ""), "score_b"] = 1.0
    _, summary = compare(pairs, fmr=0.01, system_b="visual-words")
    either = summary.set_index("method").loc["either"]
    assert either.false_match_rate == pytest.approx(1.0)


def test_calibrating_on_another_run_takes_thresholds_from_it_and_rates_from_every_evaluated_query():
    calibration, evaluation = synthetic_pairs(60, 200, seed=0), synthetic_pairs(60, 200, seed=1)
    evaluation["query_id"] = "test-" + evaluation.query_id
    # Every evaluation distractor now outscores everything: a threshold calibrated on these rows would reject them all.
    evaluation.loc[evaluation.true_id == "", ["score_a", "score_b"]] = [100.0, 100.0]
    _, summary = compare_calibrated_on(calibration, evaluation, fmr=0.05, system_b="visual-words")
    cascade = summary.set_index("method").loc["cascade"]
    expected = calibrate_cascade(calibration, fmr=0.05)
    assert (cascade.a, cascade.b) == (expected.a, expected.b)
    answer, _ = cascade_answer(evaluation, expected)
    target = evaluation.true_id != ""
    assert cascade.mean_recall == pytest.approx((answer == evaluation.true_id)[target].mean())
    assert cascade.false_match_rate == pytest.approx(1.0)
