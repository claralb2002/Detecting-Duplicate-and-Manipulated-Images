"""Own MinHash signatures and banded LSH, following Week 4 Exercises 3-7."""

from collections.abc import Sequence
import hashlib

import numpy as np

EMPTY = np.iinfo(np.uint32).max


def word_set(words: Sequence[int] | np.ndarray, vocabulary_size: int | None = None) -> np.ndarray:
    array = np.asarray(words)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError("visual words must be a one-dimensional array of integers")
    if array.size and (np.any(array < 0) or np.any(array >= (vocabulary_size if vocabulary_size is not None else EMPTY))):
        raise ValueError("visual word outside vocabulary")
    return np.unique(array.astype(np.uint32))


def jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left, right = word_set(left), word_set(right)
    intersection = len(np.intersect1d(left, right, assume_unique=True))
    union = len(left) + len(right) - intersection
    return intersection / union if union else 0.0  # Empty images are never evidence of a match.


def containment(query: Sequence[int], reference: Sequence[int]) -> float:
    query, reference = word_set(query), word_set(reference)
    return len(np.intersect1d(query, reference, assume_unique=True)) / len(query) if len(query) else 0.0


class MinHasher:
    """Independent exact random permutations over a finite visual vocabulary.

    Rank tables cost O(num_perm * vocabulary_size), manageable for <=16K words.
    Unlike affine hash approximations, these satisfy the minwise/Jaccard identity
    exactly over random construction. uint32 max is reserved for empty sets.
    """

    def __init__(self, vocabulary_size: int, *, num_perm: int = 128, seed: int = 2807):
        if type(vocabulary_size) is not int or not 1 <= vocabulary_size < EMPTY:
            raise ValueError("vocabulary_size must be a positive integer below uint32 max")
        if type(num_perm) is not int or num_perm < 1 or type(seed) is not int or seed < 0:
            raise ValueError("num_perm must be positive and seed nonnegative")
        self.vocabulary_size, self.num_perm, self.seed = vocabulary_size, num_perm, seed
        rng = np.random.default_rng(seed)
        self.ranks = np.array([rng.permutation(vocabulary_size) for _ in range(num_perm)], dtype=np.uint32)
        self.ranks.flags.writeable = False
        self.identity = hashlib.sha256(self.ranks.astype("<u4").tobytes()).hexdigest()

    def signature(self, words: Sequence[int]) -> np.ndarray:
        words = word_set(words, self.vocabulary_size)
        return self.ranks[:, words].min(axis=1) if len(words) else np.full(self.num_perm, EMPTY, np.uint32)

    def transform(self, sets: Sequence[Sequence[int]]) -> np.ndarray:
        return np.array([self.signature(s) for s in sets], dtype=np.uint32).reshape(-1, self.num_perm)


def estimated_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 1 or left.shape != right.shape or not len(left):
        raise ValueError("signatures must be nonempty vectors of equal length")
    if np.all(left == EMPTY) or np.all(right == EMPTY):
        return 0.0
    return float(np.mean(left == right))


def candidate_probability(similarity: float, b: int, r: int) -> float:
    if not np.isfinite(similarity) or not 0 <= similarity <= 1:
        raise ValueError("similarity must lie in [0, 1]")
    if any(type(v) is not int or v < 1 for v in (b, r)):
        raise ValueError("b and r must be positive integers")
    p = similarity ** r
    return float(-np.expm1(b * np.log1p(-p))) if 0 < p < 1 else p


class MinHashLSH:
    """Disjoint bands: AND r signature entries, OR b bands, with num_perm=b*r."""

    def __init__(self, signatures: np.ndarray, image_ids: Sequence[str], *, b: int, r: int):
        signatures = np.asarray(signatures)
        if any(type(v) is not int or v < 1 for v in (b, r)):
            raise ValueError("b and r must be positive integers")
        if signatures.ndim != 2 or signatures.shape != (len(image_ids), b * r) or signatures.dtype != np.uint32:
            raise ValueError("signatures must be uint32 with shape (images, b*r)")
        if len(set(image_ids)) != len(image_ids) or any(not isinstance(i, str) for i in image_ids):
            raise ValueError("image IDs must be unique strings")
        self.signatures = signatures.copy()
        self.signatures.flags.writeable = False
        self.image_ids, self.b, self.r = tuple(image_ids), b, r
        self.tables = []
        for band in range(b):
            table: dict[bytes, list[int]] = {}
            for row, signature in enumerate(signatures):
                if np.all(signature == EMPTY):
                    continue
                key = signature[band*r:(band+1)*r].astype("<u4").tobytes()
                table.setdefault(key, []).append(row)
            self.tables.append(table)

    def candidates(self, signature: np.ndarray) -> np.ndarray:
        signature = np.asarray(signature)
        if signature.shape != (self.b * self.r,) or signature.dtype != np.uint32:
            raise ValueError("query signature must be uint32 of length b*r")
        if np.all(signature == EMPTY):
            return np.empty(0, np.intp)
        rows: set[int] = set()
        for band, table in enumerate(self.tables):
            key = signature[band*self.r:(band+1)*self.r].astype("<u4").tobytes()
            rows.update(table.get(key, ()))
        return np.array(sorted(rows), dtype=np.intp)
