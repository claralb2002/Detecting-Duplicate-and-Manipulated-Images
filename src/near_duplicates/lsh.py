"""Bit sampling, band keys, one banded index for both LSH families, and their s-curves (MMDS chapter 3).

MinHash signatures come from `minhash.MinHasher`; `band_keys` turns them, like bit samples, into band keys.
"""

from __future__ import annotations

import numpy as np
import scipy.special

from near_duplicates.phash import HASH_BITS


def s_curve(similarity: np.ndarray | float, bands: int, rows: int) -> np.ndarray:
    """Probability that a pair becomes a candidate when each row agrees independently with probability `similarity`."""
    return 1.0 - (1.0 - np.asarray(similarity, dtype=float) ** rows) ** bands


def bit_sampling_curve(distance: np.ndarray | int, bands: int, rows: int, n_bits: int = HASH_BITS) -> np.ndarray:
    """Exact candidate probability for two codes `distance` bits apart under `BitSampler`.

    A band draws its `rows` positions without replacement, so it agrees with probability
    C(n - d, r) / C(n, r), slightly below the (1 - d/n)^r that `s_curve` assumes.
    """
    distance = np.asarray(distance)
    per_band = scipy.special.comb(n_bits - distance, rows) / scipy.special.comb(n_bits, rows)
    return 1.0 - (1.0 - per_band) ** bands



def band_keys(signatures: np.ndarray, bands: int, rows: int, seed: int = 0) -> np.ndarray:
    """Collapse each band of `rows` signature values into one 64-bit key, shape (n, bands).

    The mixers are odd, hence invertible modulo 2^64, so changing any single value changes the key.
    """
    n, width = signatures.shape
    if bands * rows > width:
        raise ValueError(f"{bands} bands x {rows} rows needs {bands * rows} values, signature has {width}")
    mixers = np.random.default_rng(seed).integers(1, 2**63, size=rows, dtype=np.uint64) | np.uint64(1)
    blocks = signatures[:, : bands * rows].astype(np.uint64).reshape(n, bands, rows)
    with np.errstate(over="ignore"):
        return (blocks * mixers).sum(axis=2, dtype=np.uint64)


class BitSampler:
    """LSH for Hamming distance on 64-bit codes: each band reads `rows` distinct random bit positions."""

    def __init__(self, bands: int, rows: int, n_bits: int = HASH_BITS, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.positions = np.stack([rng.choice(n_bits, size=rows, replace=False) for _ in range(bands)])
        self.bands, self.rows, self.n_bits = bands, rows, n_bits

    def keys(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes, dtype=np.uint64)
        bits = (codes[:, None, None] >> self.positions.astype(np.uint64)[None]) & np.uint64(1)
        weights = np.uint64(1) << np.arange(self.rows, dtype=np.uint64)
        return (bits * weights).sum(axis=2, dtype=np.uint64)


BAND_MIXER = np.uint64(0x9E3779B97F4A7C15)


def band_tagged(keys: np.ndarray) -> np.ndarray:
    """Fold the band number into each key, so all bands share one sorted table and one lookup.

    Equal (band, key) pairs always give equal tagged keys, so no true candidate is lost; a rare
    collision between different pairs only adds a spurious candidate, which verification rejects.
    """
    with np.errstate(over="ignore"):
        return keys.astype(np.uint64) * BAND_MIXER + np.arange(keys.shape[-1], dtype=np.uint64)


class BandIndex:
    """All band keys in one sorted table; a query returns every item sharing at least one band key."""

    def __init__(self, keys: np.ndarray):
        n, self.bands = keys.shape
        tagged = band_tagged(keys).ravel()
        order = np.argsort(tagged, kind="stable")
        self.sorted_keys = tagged[order]
        self.items = (order // self.bands).astype(np.int32 if n < 2**31 else np.int64)

    def candidates(self, query_keys: np.ndarray) -> np.ndarray:
        """Sorted candidate item positions for one query, given its keys of shape (bands,)."""
        tagged = band_tagged(np.asarray(query_keys)[None])[0]
        starts = np.searchsorted(self.sorted_keys, tagged, side="left")
        counts = np.searchsorted(self.sorted_keys, tagged, side="right") - starts
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        offsets = np.cumsum(counts) - counts
        return np.unique(self.items[np.arange(total) + np.repeat(starts - offsets, counts)]).astype(np.int64)

    @property
    def nbytes(self) -> int:
        return self.sorted_keys.nbytes + self.items.nbytes


