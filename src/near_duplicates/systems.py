"""The two retrieval systems behind one interface, and the search loop that writes the shared results rows.

System A ("phash"): 64-bit perceptual hash, bit-sampling LSH, score = -Hamming distance.
System B ("visual-words"): SIFT visual-word sets, MinHash with banded LSH, score = Jaccard similarity.
Each system is searched exhaustively and through its LSH index, so the loss caused by LSH is measurable.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np

from near_duplicates.features import FeatureSet
from near_duplicates.lsh import CODE_BITS, BandIndex, BitSampler, MinHasher, band_keys, hamming
from near_duplicates.vocabulary import incidence, word_sets


class System(Protocol):
    name: str
    index: BandIndex

    def encode(self, features: FeatureSet) -> Any: ...
    def keys(self, encoded: Any) -> np.ndarray: ...
    def scores(self, encoded: Any, query: int) -> np.ndarray: ...
    def similarity(self, score: float) -> float: ...


class PhashSystem:
    name = "phash"

    def __init__(self, hashes: np.ndarray, bands: int, rows: int, seed: int):
        self.hashes = hashes
        self.sampler = BitSampler(bands, rows, seed=seed)
        self.index = BandIndex(self.sampler.keys(hashes))

    def encode(self, features: FeatureSet) -> np.ndarray:
        return features.hashes

    def keys(self, hashes: np.ndarray) -> np.ndarray:
        return self.sampler.keys(hashes)

    def scores(self, hashes: np.ndarray, query: int) -> np.ndarray:
        return -hamming(self.hashes, hashes[query]).astype(float)

    def similarity(self, score: float) -> float:
        """Fraction of agreeing bits, the per-row agreement probability of bit sampling."""
        return 1 + score / CODE_BITS


class VisualWordSystem:
    name = "visual-words"

    def __init__(self, descriptors: list[np.ndarray], centers: np.ndarray, num_perm: int, bands: int, rows: int, seed: int):
        self.centers, self.bands, self.rows, self.seed = centers, bands, rows, seed
        sets = word_sets(descriptors, centers)
        self.matrix = incidence(sets, len(centers))
        self.sizes = np.asarray(self.matrix.sum(axis=1)).ravel()
        self.minhasher = MinHasher(num_perm, seed=seed)
        self.index = BandIndex(self.keys(sets))

    def encode(self, features: FeatureSet) -> list[np.ndarray]:
        return word_sets(features.descriptors, self.centers)

    def keys(self, sets: list[np.ndarray]) -> np.ndarray:
        return band_keys(self.minhasher.signatures(sets), self.bands, self.rows, seed=self.seed + 1)

    def scores(self, sets: list[np.ndarray], query: int) -> np.ndarray:
        indicator = np.zeros(len(self.centers), np.float32)
        indicator[sets[query]] = 1
        intersection = self.matrix @ indicator
        return intersection / np.maximum(self.sizes + len(sets[query]) - intersection, 1)

    def similarity(self, score: float) -> float:
        return score


def search(systems: list[System], index_ids: list[str], queries: FeatureSet, true_ids: list[str],
           kind: str, strength: float) -> list[dict]:
    """Rows in the shared results schema for every query, system and mode (exhaustive and LSH)."""
    position = {image_id: i for i, image_id in enumerate(index_ids)}
    everything = np.arange(len(index_ids))
    rows = []
    for system in systems:
        encoded = system.encode(queries)
        keys = system.keys(encoded)
        for i, (query_id, true_id) in enumerate(zip(queries.ids, true_ids)):
            scores = system.scores(encoded, i)
            start = time.perf_counter()
            candidates = system.index.candidates(keys[i])
            lsh_seconds = time.perf_counter() - start
            true_position = position[true_id] if true_id else -1
            for mode, pool in (("exhaustive", everything), ("lsh", candidates)):
                best = int(pool[np.argmax(scores[pool])]) if len(pool) else -1
                rows.append(dict(
                    system=system.name, mode=mode, kind=kind, strength=strength, query_id=query_id, true_id=true_id,
                    predicted=index_ids[best] if best >= 0 else "",
                    score=float(scores[best]) if best >= 0 else -np.inf,
                    candidates=len(pool),
                    true_is_candidate=bool(true_position >= 0 and np.isin(true_position, candidates)),
                    true_similarity=system.similarity(scores[true_position]) if true_position >= 0 else np.nan,
                    lsh_seconds=lsh_seconds if mode == "lsh" else np.nan,
                ))
    return rows
