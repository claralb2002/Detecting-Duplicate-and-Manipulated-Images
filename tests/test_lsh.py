from __future__ import annotations

import numpy as np
import pytest

from near_duplicates.lsh import BandIndex, BitSampler, band_keys, bit_sampling_curve, s_curve


def random_codes(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2**64, size=n, dtype=np.uint64)


def test_bit_sampler_key_depends_only_on_sampled_bits():
    sampler = BitSampler(bands=4, rows=8, seed=0)
    code = random_codes(1, 2)[0]
    sampled = set(sampler.positions[0].tolist())
    outside = next(bit for bit in range(64) if bit not in sampled)
    inside = sampler.positions[0][3]
    flip = lambda bit: np.array([code ^ (np.uint64(1) << np.uint64(bit))], dtype=np.uint64)
    base = sampler.keys(np.array([code]))[0, 0]
    assert sampler.keys(flip(outside))[0, 0] == base
    assert sampler.keys(flip(inside))[0, 0] != base


@pytest.mark.parametrize("distance", [4, 12, 24])
def test_bit_sampling_curve_matches_simulation(distance):
    """Over random samplers, the candidate rate at a fixed distance follows the exact hypergeometric curve."""
    rng = np.random.default_rng(distance)
    bands, rows, trials = 4, 8, 3000
    hits = 0
    for trial in range(trials):
        a = random_codes(1, trial)[0]
        mask = np.uint64(sum(1 << int(bit) for bit in rng.choice(64, size=distance, replace=False)))
        sampler = BitSampler(bands, rows, seed=trial)
        keys = sampler.keys(np.array([a, a ^ mask], dtype=np.uint64))
        hits += bool((keys[0] == keys[1]).any())
    expected = float(bit_sampling_curve(distance, bands, rows))
    standard_error = np.sqrt(expected * (1 - expected) / trials)
    assert abs(hits / trials - expected) < 4 * standard_error + 1e-3
    assert expected < float(s_curve(1 - distance / 64, bands, rows))  # sampling without replacement is stricter


def test_band_key_changes_when_any_single_value_changes():
    signatures = np.random.default_rng(0).integers(0, 2**31, size=(1, 12), dtype=np.uint64)
    base = band_keys(signatures, bands=3, rows=4)
    for column in range(12):
        changed = signatures.copy()
        changed[0, column] += np.uint64(1)
        keys = band_keys(changed, bands=3, rows=4)
        band = column // 4
        assert keys[0, band] != base[0, band]
        assert (np.delete(keys[0], band) == np.delete(base[0], band)).all()


def test_band_keys_rejects_too_short_signatures():
    with pytest.raises(ValueError):
        band_keys(np.zeros((2, 10), dtype=np.uint64), bands=4, rows=3)


def test_band_index_returns_exactly_the_items_sharing_a_band_key():
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 6, size=(400, 5)).astype(np.uint64)  # small alphabet forces many collisions
    index = BandIndex(keys)
    for _ in range(50):
        query = rng.integers(0, 8, size=5).astype(np.uint64)
        expected = np.flatnonzero((keys == query).any(axis=1))
        assert index.candidates(query).tolist() == expected.tolist()
