from __future__ import annotations

import numpy as np
import pytest

from near_duplicates.lsh import BandIndex, BitSampler, MinHasher, band_keys, bit_sampling_curve, s_curve
from near_duplicates.tune import JACCARD_VALUES, hamming_profile, jaccard_bins, jaccard_profile
from near_duplicates.vocabulary import incidence


def overlapping_sets(rng: np.random.Generator, n: int, vocabulary: int) -> list[np.ndarray]:
    """Sets drawn from a few shared pools, so Jaccard similarities spread from 0 to well above 0.5."""
    pools = [rng.choice(vocabulary, size=60, replace=False) for _ in range(5)]
    return [np.unique(np.concatenate([rng.choice(pools[i % 5], size=rng.integers(10, 50), replace=False),
                                      rng.choice(vocabulary, size=5, replace=False)])) for i in range(n)]


def test_jaccard_bins_keep_exact_zeros_apart_and_put_one_in_the_top_bin():
    bins = jaccard_bins(np.array([0.0, 1e-6, 0.5, 1.0]))
    assert bins.tolist() == [0, 1, 5001, 10000]
    assert JACCARD_VALUES[bins[0]] == 0 and JACCARD_VALUES[bins[-1]] == pytest.approx(1, abs=1e-4)


def test_jaccard_profile_matches_set_arithmetic():
    rng = np.random.default_rng(0)
    sets = overlapping_sets(rng, 60, 300)
    index, queries = incidence(sets[:50], 300).toarray(), incidence(sets[50:], 300).toarray()
    true_positions = np.array([3, 7, 7, 0, 49, -1, -1, 12, 20, 33])
    profile = jaccard_profile(index, index.sum(axis=1), queries, true_positions)
    jaccard = np.array([[len(np.intersect1d(q, r)) / len(np.union1d(q, r)) for r in sets[:50]] for q in sets[50:]])
    for i, t in enumerate(true_positions):
        if t >= 0:
            assert profile.true_similarity[i] == pytest.approx(jaccard[i, t], abs=1e-6)
            assert profile.rank[i] == (jaccard[i] > jaccard[i, t] + 1e-6).sum()
        assert profile.top_score[i] == pytest.approx(jaccard[i].max(), abs=1e-6)
    assert profile.histogram.sum() == jaccard.size and profile.histogram[0] == (jaccard == 0).sum()


def test_predicted_minhash_candidate_fraction_matches_a_real_band_index():
    rng = np.random.default_rng(1)
    sets = overlapping_sets(rng, 400, 500)
    index_sets, query_sets = sets[:300], sets[300:]
    matrix = incidence(index_sets, 500).toarray()
    profile = jaccard_profile(matrix, matrix.sum(axis=1), incidence(query_sets, 500).toarray(), np.full(100, -1))
    bands, rows = 20, 3
    measured = []
    for seed in range(5):
        hasher = MinHasher(bands * rows, seed=seed)
        index = BandIndex(band_keys(hasher.signatures(index_sets), bands, rows, seed=seed))
        keys = band_keys(hasher.signatures(query_sets), bands, rows, seed=seed)
        measured.append(np.mean([len(index.candidates(k)) for k in keys]) / len(index_sets))
    predicted = profile.candidate_fraction(lambda s: s_curve(s, bands, rows))
    assert 0.05 < predicted < 0.95
    assert np.mean(measured) == pytest.approx(predicted, rel=0.05)


def test_predicted_bit_sampling_candidate_fraction_matches_a_real_band_index():
    rng = np.random.default_rng(2)
    base = rng.integers(0, 2**64, size=20, dtype=np.uint64)
    flips = lambda n: np.array([np.uint64(sum(1 << int(b) for b in rng.choice(64, size=rng.integers(0, 20), replace=False)))
                                for _ in range(n)], dtype=np.uint64)
    index_codes = np.repeat(base, 25) ^ flips(500)
    query_codes = base[rng.integers(0, 20, size=100)] ^ flips(100)
    profile = hamming_profile(index_codes, query_codes, np.full(100, -1))
    bands, rows = 8, 6
    measured = []
    for seed in range(20):
        sampler = BitSampler(bands, rows, seed=seed)
        index = BandIndex(sampler.keys(index_codes))
        measured.append(np.mean([len(index.candidates(k)) for k in sampler.keys(query_codes)]) / len(index_codes))
    predicted = profile.candidate_fraction(lambda d: bit_sampling_curve(d, bands, rows))
    assert 0.02 < predicted < 0.5
    assert np.mean(measured) == pytest.approx(predicted, rel=0.05)


def test_hamming_profile_ranks_the_true_code_and_finds_the_nearest():
    index = np.array([0b0, 0b1111, 0b11111111, 0b1], dtype=np.uint64)
    queries = np.array([0b11, 0b11111110], dtype=np.uint64)
    profile = hamming_profile(index, queries, np.array([1, 2]))
    assert profile.true_similarity.tolist() == [-2, -1]
    assert profile.rank.tolist() == [1, 0]  # query 0: only code 0b1 (distance 1) beats the true code (distance 2)
    assert profile.top_correct.tolist() == [False, True]
    assert profile.histogram.sum() == 8
