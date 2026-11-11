"""Bit-sampling LSH with independent tables and exact candidate verification.

Each table samples r distinct positions uniformly from 64 bits. Tables are
independent and may overlap; b*r need not equal 64. AND within each table,
OR across tables, then verify the deduplicated union with full Hamming distance.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

import numpy as np

from near_duplicates.phash import HASH_BITS, hash_array, hash_value, hamming_distances


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
