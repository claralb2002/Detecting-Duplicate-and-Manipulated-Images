from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from near_duplicates.metrics import (
    RESULT_COLUMNS,
    calibrate,
    micro_average_precision,
    percentile_interval,
    pooled_rates,
    retrieval_scores,
    source_resamples,
    summarize,
    threshold_at_fmr,
    wilson_interval,
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


def test_lsh_loss_predicts_from_true_similarity_and_exhaustive_hits():
    from near_duplicates.lsh import bit_sampling_curve, s_curve
    from near_duplicates.metrics import lsh_loss

    def pair(system, query_id, exhaustive_predicted, lsh_predicted, similarity, candidate):
        base = dict(system=system, kind="crop", strength=0.5, query_id=query_id, true_id="t" + query_id, score=1.0,
                    candidates=1, true_similarity=similarity, seconds=0.0)
        return [dict(base, mode="exhaustive", predicted=exhaustive_predicted, true_is_candidate=candidate),
                dict(base, mode="lsh", predicted=lsh_predicted, true_is_candidate=candidate)]
    rows = (pair("phash", "1", "t1", "t1", 1 - 20 / 64, True) + pair("phash", "2", "x", "", 1 - 40 / 64, False)
            + pair("visual-words", "1", "t1", "", 0.2, False) + pair("visual-words", "2", "t2", "t2", 0.6, True))
    rows.append(dict(rows[0], query_id="d", true_id="", predicted="y", true_similarity=np.nan))  # distractors are ignored
    loss = lsh_loss(pd.DataFrame(rows, columns=RESULT_COLUMNS), 16, 8, 64, 2).set_index("system")
    phash, words = loss.loc["phash"], loss.loc["visual-words"]
    assert phash.queries == 2 and phash.exhaustive_recall == 0.5 and phash.lsh_recall == 0.5
    assert phash.predicted_lsh_recall == pytest.approx(bit_sampling_curve(20, 16, 8) / 2)
    assert phash.predicted_candidate_rate == pytest.approx((bit_sampling_curve(20, 16, 8) + bit_sampling_curve(40, 16, 8)) / 2)
    assert words.predicted_lsh_recall == pytest.approx((s_curve(0.2, 64, 2) + s_curve(0.6, 64, 2)) / 2)
    assert words.candidate_rate == 0.5


def test_source_bootstrap_widens_the_interval_only_when_rows_of_a_source_agree():
    # 500 sources; 25 of them (5%) match falsely. Independent rows: one row per source.
    # Clustered rows: each source repeated under 50 conditions with the same outcome, as a near-duplicate source behaves.
    outcome = np.zeros(500)
    outcome[:25] = 1
    sources = np.arange(500)
    one = source_resamples(sources, np.column_stack([outcome, np.ones(500)]), seed=0)
    low, high = percentile_interval(one[:, 0] / one[:, 1])
    wilson_low, wilson_high = wilson_interval(25, 500)
    assert high - low == pytest.approx(wilson_high - wilson_low, rel=0.2)
    repeated = source_resamples(np.repeat(sources, 50), np.column_stack([np.repeat(outcome, 50), np.ones(25_000)]), seed=0)
    low_r, high_r = percentile_interval(repeated[:, 0] / repeated[:, 1])
    naive_low, naive_high = wilson_interval(1250, 25_000)
    assert high_r - low_r == pytest.approx(high - low, rel=0.2)  # 50 copies carry no more information than one
    assert high_r - low_r > 4 * (naive_high - naive_low)


def test_pooled_rates_count_accepted_hits_and_false_matches_over_all_conditions():
    rows = [row("s", "lsh", kind, 1, f"t{i}", f"t{i}", f"t{i}" if i < 3 else "x", 0.9) for kind in ("a", "b") for i in range(4)]
    rows += [row("s", "lsh", kind, 1, f"d{i}", "", "y", 0.9 if i == 0 else 0.1) for kind in ("a", "b") for i in range(5)]
    pooled = pooled_rates(pd.DataFrame(rows, columns=RESULT_COLUMNS), pd.DataFrame({"system": ["s"], "mode": ["lsh"], "threshold": [0.5]}))
    result = pooled.iloc[0]
    assert result.recall_at_threshold == 0.75 and result.false_match_rate == 0.2
    assert result.recall_low <= 0.75 <= result.recall_high and result.fmr_low <= 0.2 <= result.fmr_high
