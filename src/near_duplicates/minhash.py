"""Own MinHash signatures, following Week 4 Exercises 3-7; banding is `lsh.band_keys` with `lsh.BandIndex`."""

import hashlib
from collections.abc import Sequence

import numpy as np

EMPTY = np.iinfo(np.uint32).max


def word_set(words: Sequence[int] | np.ndarray, vocabulary_size: int | None = None) -> np.ndarray:
    array = np.asarray(words)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError("visual words must be a one-dimensional array of integers")
    if array.size and (np.any(array < 0) or np.any(array >= (vocabulary_size if vocabulary_size is not None else EMPTY))):
        raise ValueError("visual word outside vocabulary")
    return np.unique(array.astype(np.uint32))



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
