from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from near_duplicates.metrics import (
    RESULT_COLUMNS, calibrate, micro_average_precision, retrieval_scores, summarize, threshold_at_fmr, wilson_interval,
)


def test_wilson_interval_known_values():
    low, high = wilson_interval(100, 200)
    assert low == pytest.approx(0.5 - 0.0686, abs=1e-4) and high == pytest.approx(0.5 + 0.0686, abs=1e-4)
    low, high = wilson_interval(0, 200)
    assert low == pytest.approx(0, abs=1e-12) and high == pytest.approx(0.01885, abs=1e-4)  # non-zero width at 0 successes


def test_threshold_respects_target_rate_with_ties():
    scores = np.array([-10] * 50 + [-3] * 30 + [-2] * 15 + [-1] * 5, dtype=float)  # integer pHash-like scores
    for fmr in (0.01, 0.05, 0.1, 0.2, 0.5):
        threshold = threshold_at_fmr(scores, fmr)
        assert (scores > threshold).mean() <= fmr
        lower = scores[scores < threshold]
        if len(lower):  # the threshold is the loosest one that still meets the rate
            assert (scores > lower.max()).mean() > fmr


def test_threshold_on_continuous_scores_hits_the_rate_exactly():
    scores = np.random.default_rng(0).random(1000)
    assert (scores > threshold_at_fmr(scores, 0.01)).sum() == 10


def row(system, mode, kind, strength, query_id, true_id, predicted, score) -> dict:
    return dict(system=system, mode=mode, kind=kind, strength=strength, query_id=query_id, true_id=true_id,
                predicted=predicted, score=score, candidates=1, true_is_candidate=True, true_similarity=0.5, seconds=0.0)


def test_summary_counts_hits_thresholds_and_false_matches():
    rows = [
        row("s", "lsh", "crop", 0.5, "q1", "a", "a", 0.9),   # correct, accepted
        row("s", "lsh", "crop", 0.5, "q2", "b", "b", 0.2),   # correct, below threshold
        row("s", "lsh", "crop", 0.5, "q3", "c", "x", 0.95),  # wrong, so not a hit at any threshold
        row("s", "lsh", "crop", 0.5, "q4", "d", "", -np.inf),  # no candidates
        row("s", "lsh", "crop", 0.5, "d1", "", "y", 0.7),    # distractor accepted: false match
        row("s", "lsh", "crop", 0.5, "d2", "", "z", 0.1),
    ]
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    summary = summarize(results, pd.DataFrame({"system": ["s"], "mode": ["lsh"], "threshold": [0.5]})).iloc[0]
    assert summary.queries == 4 and summary.recall_at_1 == 0.5 and summary.recall_at_threshold == 0.25
    assert summary.false_match_rate == 0.5
    assert summary.recall_at_1_low < 0.5 < summary.recall_at_1_high


def test_calibration_uses_only_distractors():
    rows = [row("s", "lsh", "jpeg", 10, f"t{i}", "a", "a", 100.0) for i in range(5)]
    rows += [row("s", "lsh", "jpeg", 10, f"d{i}", "", "y", float(i)) for i in range(100)]
    thresholds = calibrate(pd.DataFrame(rows, columns=RESULT_COLUMNS), fmr=0.05)
    assert thresholds.threshold.item() == 94.0


def test_micro_average_precision_hand_computed():
    # Ranked: hit, miss, hit; 3 positive queries, so one positive is never found.
    average_precision, recall_at_p90 = micro_average_precision([True, False, True], [0.9, 0.8, 0.7], positives=3)
    assert average_precision == pytest.approx(1 / 3 * 1 + 1 / 3 * 2 / 3)
    assert recall_at_p90 == pytest.approx(1 / 3)


def test_micro_average_precision_ranks_ties_worst_case_like_the_official_code():
    assert micro_average_precision([True, False], [0.5, 0.5], positives=1)[0] == pytest.approx(0.5)
    assert micro_average_precision([False, True], [0.5, 0.5], positives=1)[0] == pytest.approx(0.5)
    assert micro_average_precision([False, True], [0.4, 0.5], positives=1)[0] == pytest.approx(1.0)


def test_retrieval_scores_ignore_queries_without_prediction_but_count_their_positives():
    rows = [row("s", "lsh", "dev_report", 0, "q1", "a", "a", 0.9), row("s", "lsh", "dev_report", 0, "q2", "b", "", -np.inf),
            row("s", "lsh", "dev_report", 0, "q3", "", "z", 0.1)]
    scores = retrieval_scores(pd.DataFrame(rows, columns=RESULT_COLUMNS)).iloc[0]
    assert scores.micro_ap == pytest.approx(0.5) and scores.recall_at_p90 == pytest.approx(0.5)
