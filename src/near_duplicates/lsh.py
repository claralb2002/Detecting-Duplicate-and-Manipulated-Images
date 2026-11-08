"""MinHash, bit-sampling LSH, band keys and a banded index over sorted arrays (MMDS chapter 3).

`BitSampler` with `BandIndex` serves the experiment pipeline. `BitSamplingLSH`, the System A
calibration index (`system_a.py`), is a bit-sampling LSH with independent tables and exact candidate
verification. Each table samples r distinct positions uniformly from 64 bits. Tables are
independent and may overlap; b*r need not equal 64. AND within each table,
OR across tables, then verify the deduplicated union with full Hamming distance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

import numpy as np
import scipy.special

from near_duplicates.phash import HASH_BITS, hash_array, hash_value, hamming_distances

MERSENNE_PRIME = np.uint64((1 << 31) - 1)
CODE_BITS = 64


def s_curve(similarity: np.ndarray | float, bands: int, rows: int) -> np.ndarray:
    """Probability that a pair becomes a candidate when each row agrees independently with probability `similarity`."""
    return 1.0 - (1.0 - np.asarray(similarity, dtype=float) ** rows) ** bands


def bit_sampling_curve(distance: np.ndarray | int, bands: int, rows: int, n_bits: int = CODE_BITS) -> np.ndarray:
    """Exact candidate probability for two codes `distance` bits apart under `BitSampler`.

    A band draws its `rows` positions without replacement, so it agrees with probability
    C(n - d, r) / C(n, r), slightly below the (1 - d/n)^r that `s_curve` assumes.
    """
    distance = np.asarray(distance)
    per_band = scipy.special.comb(n_bits - distance, rows) / scipy.special.comb(n_bits, rows)
    return 1.0 - (1.0 - per_band) ** bands


class MinHasher:
    """Min-wise hashing with universal hash functions h(x) = (a x + b) mod p."""

    def __init__(self, num_perm: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, MERSENNE_PRIME, size=(num_perm, 1), dtype=np.uint64)
        self.b = rng.integers(0, MERSENNE_PRIME, size=(num_perm, 1), dtype=np.uint64)

    def signature(self, items: np.ndarray) -> np.ndarray:
        if len(items) == 0:
            return np.full(self.num_perm, MERSENNE_PRIME, dtype=np.uint64)
        x = np.asarray(items, dtype=np.uint64)[None, :] % MERSENNE_PRIME
        return ((self.a * x + self.b) % MERSENNE_PRIME).min(axis=1)

    def signatures(self, sets: list[np.ndarray]) -> np.ndarray:
        return np.stack([self.signature(s) for s in sets])


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

    def __init__(self, bands: int, rows: int, n_bits: int = CODE_BITS, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.positions = np.stack([rng.choice(n_bits, size=rows, replace=False) for _ in range(bands)])
        self.bands, self.rows, self.n_bits = bands, rows, n_bits

    def keys(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes, dtype=np.uint64)
        bits = (codes[:, None, None] >> self.positions.astype(np.uint64)[None]) & np.uint64(1)
        weights = np.uint64(1) << np.arange(self.rows, dtype=np.uint64)
        return (bits * weights).sum(axis=2, dtype=np.uint64)


class BandIndex:
    """One sorted key table per band; a query returns every item sharing at least one band key."""

    def __init__(self, keys: np.ndarray):
        self.order = np.argsort(keys, axis=0, kind="stable")
        self.sorted_keys = np.take_along_axis(keys, self.order, axis=0)

    def candidates(self, query_keys: np.ndarray) -> np.ndarray:
        """Sorted candidate item positions for one query, given its keys of shape (bands,)."""
        found = []
        for band, key in enumerate(query_keys):
            column = self.sorted_keys[:, band]
            start, stop = np.searchsorted(column, key, side="left"), np.searchsorted(column, key, side="right")
            if stop > start:
                found.append(self.order[start:stop, band])
        return np.unique(np.concatenate(found)) if found else np.empty(0, dtype=np.int64)


def hamming(codes: np.ndarray, code: int | np.uint64) -> np.ndarray:
    return np.bitwise_count(np.asarray(codes, dtype=np.uint64) ^ np.uint64(code))


def _positive_integer(name: str, value: int, maximum: int | None = None) -> None:
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be a positive integer" + (f" <= {maximum}" if maximum else ""))


def _radius(value: int) -> None:
    if type(value) is not int or not 0 <= value <= HASH_BITS:
        raise ValueError("max_distance must be an integer in [0, 64]")


def collision_probability(distance: int, b: int, r: int) -> float:
    """Exact retrieval probability over random tables for a pair at distance d.

    P(table collision) = C(64-d,r)/C(64,r), since sampling is without replacement.
    1-(1-(1-d/64)**r)**b is the usual with-replacement approximation, not the
    exact law for these tables. Probabilities describe random index construction,
    not a guarantee for any particular pair in one fixed index.
    """
    _radius(distance)
    _positive_integer("b", b)
    _positive_integer("r", r, HASH_BITS)
    p = comb(HASH_BITS - distance, r) / comb(HASH_BITS, r) if r <= HASH_BITS - distance else 0.0
    return float(-np.expm1(b * np.log1p(-p))) if 0 < p < 1 else p


@dataclass(frozen=True)
class Match:
    image_id: str
    distance: int


def verify_candidates(
    query: int, hashes: Sequence[int] | np.ndarray, image_ids: Sequence[str],
    candidates: Sequence[int] | np.ndarray, *, max_distance: int, top_k: int | None = 1,
) -> list[Match]:
    """Deduplicate, verify, threshold inclusively and rank by (distance, image ID).

    An empty list is a valid no-match decision. top_k=None returns all accepted
    candidates. Catalogue rows are explicit so duplicate hashes retain all IDs.
    """
    _radius(max_distance)
    if top_k is not None:
        _positive_integer("top_k", top_k)
    hashes = hash_array(hashes)
    if len(hashes) != len(image_ids):
        raise ValueError("image_ids and hashes must have equal lengths")
    rows = np.asarray(candidates)
    if rows.ndim != 1 or (rows.size and rows.dtype.kind not in "iu"):
        raise ValueError("candidates must be one-dimensional integer row indices")
    if rows.size and (np.any(rows < 0) or np.any(rows >= len(hashes))):
        raise ValueError("candidate row outside catalogue")
    rows = np.unique(rows.astype(np.intp))
    distances = hamming_distances(query, hashes[rows])
    matches = sorted(
        (Match(image_ids[int(row)], int(d)) for row, d in zip(rows, distances) if d <= max_distance),
        key=lambda match: (match.distance, match.image_id),
    )
    return matches if top_k is None else matches[:top_k]


class BitSamplingLSH:
    """Read-only catalogue with O(b*N) bucket entries and eight-byte pHashes."""

    def __init__(
        self, hashes: Sequence[int] | np.ndarray, image_ids: Sequence[str], *,
        b: int = 16, r: int = 12, seed: int = 2807,
    ):
        _positive_integer("b", b)
        _positive_integer("r", r, HASH_BITS)
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.hashes = hash_array(hashes).copy()
        self.image_ids = tuple(image_ids)
        if len(self.hashes) != len(self.image_ids) or len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("provide one unique image ID per hash")
        if any(not isinstance(i, str) for i in self.image_ids):
            raise ValueError("image IDs must be strings")
        self.hashes.flags.writeable = False
        self.b, self.r, self.seed = b, r, seed
        rng = np.random.default_rng(seed)
        self.positions = np.array([rng.choice(HASH_BITS, size=r, replace=False) for _ in range(b)])
        self.positions.flags.writeable = False
        self.masks = tuple(np.uint64(sum(1 << int(p) for p in row)) for row in self.positions)
        self._tables: list[dict[int, list[int]]] = []
        for mask in self.masks:
            buckets: dict[int, list[int]] = {}
            for row, key in enumerate(self.hashes & mask):
                buckets.setdefault(int(key), []).append(row)
            self._tables.append(buckets)

    def candidates(self, query: int) -> np.ndarray:
        """Return sorted, unique catalogue row indices; zero collisions means empty."""
        query = hash_value(query)
        rows: set[int] = set()
        for mask, buckets in zip(self.masks, self._tables):
            rows.update(buckets.get(query & int(mask), ()))
        return np.array(sorted(rows), dtype=np.intp)

    def query(self, query: int, *, max_distance: int, top_k: int | None = 1) -> list[Match]:
        return verify_candidates(
            query, self.hashes, self.image_ids, self.candidates(query),
            max_distance=max_distance, top_k=top_k,
        )
