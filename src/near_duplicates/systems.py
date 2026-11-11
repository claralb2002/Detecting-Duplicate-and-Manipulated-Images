"""The two retrieval systems behind one interface, and the search loop that writes the shared results rows.

System A ("phash"): 64-bit perceptual hash, bit-sampling LSH, score = -Hamming distance.
System B ("visual-words"): SIFT visual-word sets, MinHash with banded LSH, score = Jaccard similarity.
System B with spatial verification ("visual-words+ransac"): B's top-k by Jaccard re-ranked by RANSAC inliers.
System B scored by containment ("visual-words-containment"): the share of the query's words found in the image.
Each system is searched exhaustively and through its LSH index, so the loss caused by LSH is measurable.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np

from near_duplicates.features import FeatureSet
from near_duplicates.lsh import CODE_BITS, BandIndex, BitSampler, MinHasher, band_keys, hamming
from near_duplicates.verification import inliers
from near_duplicates.vocabulary import incidence, word_sets


class System(Protocol):
    name: str
    index: BandIndex

    def encode(self, features: FeatureSet) -> Any: ...
    def keys(self, encoded: Any) -> np.ndarray: ...
    def rank(self, encoded: Any, query: int, positions: np.ndarray | None = None) -> tuple[int, float]: ...
    def similarity(self, encoded: Any, query: int, position: int) -> float: ...
    def memory_bytes(self) -> int: ...


def top(scores: np.ndarray, positions: np.ndarray | None) -> tuple[int, float]:
    """Index position and score of the best of `scores`, which were computed for `positions` (all if None)."""
    best = int(np.argmax(scores))
    return (best if positions is None else int(positions[best])), float(scores[best])


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

    def scores(self, hashes: np.ndarray, query: int, positions: np.ndarray | None = None) -> np.ndarray:
        """Scores against the indexed images at `positions`, or against all of them."""
        codes = self.hashes if positions is None else self.hashes[positions]
        return -hamming(codes, hashes[query]).astype(float)

    def rank(self, hashes: np.ndarray, query: int, positions: np.ndarray | None = None) -> tuple[int, float]:
        return top(self.scores(hashes, query, positions), positions)

    def similarity(self, hashes: np.ndarray, query: int, position: int) -> float:
        """Fraction of agreeing bits, the per-row agreement probability of bit sampling."""
        return 1 + float(self.scores(hashes, query, np.array([position]))[0]) / CODE_BITS

    def memory_bytes(self) -> int:
        return self.hashes.nbytes + self.index.nbytes


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

    def intersections(self, sets: list[np.ndarray], query: int, positions: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Words shared with, and word counts of, the indexed images at `positions`, or of all of them."""
        matrix, sizes = (self.matrix, self.sizes) if positions is None else (self.matrix[positions], self.sizes[positions])
        indicator = np.zeros(len(self.centers), np.float32)
        indicator[sets[query]] = 1
        return matrix @ indicator, sizes

    def scores(self, sets: list[np.ndarray], query: int, positions: np.ndarray | None = None) -> np.ndarray:
        """Jaccard similarity against the indexed images at `positions`, or against all of them."""
        intersection, sizes = self.intersections(sets, query, positions)
        return intersection / np.maximum(sizes + len(sets[query]) - intersection, 1)

    def rank(self, sets: list[np.ndarray], query: int, positions: np.ndarray | None = None) -> tuple[int, float]:
        return top(self.scores(sets, query, positions), positions)

    def similarity(self, sets: list[np.ndarray], query: int, position: int) -> float:
        """Jaccard similarity, the per-row agreement probability of MinHash."""
        return float(self.scores(sets, query, np.array([position]))[0])

    def memory_bytes(self) -> int:
        matrix = self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes
        return matrix + self.sizes.nbytes + self.centers.nbytes + self.index.nbytes


class ContainmentSystem:
    """System B's retrieval and LSH, scored by containment |Q intersect X| / |Q| instead of Jaccard similarity.

    A crop keeps a subset of its source's words, so its Jaccard similarity falls with the area cropped away
    while its containment stays high. An indexed image with many words contains more of any query by chance,
    which is what the threshold on distractors has to absorb.
    """

    name = "visual-words-containment"

    def __init__(self, base: VisualWordSystem):
        self.base, self.index = base, base.index

    def encode(self, features: FeatureSet) -> list[np.ndarray]:
        return self.base.encode(features)

    def keys(self, sets: list[np.ndarray]) -> np.ndarray:
        return self.base.keys(sets)

    def scores(self, sets: list[np.ndarray], query: int, positions: np.ndarray | None = None) -> np.ndarray:
        intersection, _ = self.base.intersections(sets, query, positions)
        return intersection / max(len(sets[query]), 1)

    def rank(self, sets: list[np.ndarray], query: int, positions: np.ndarray | None = None) -> tuple[int, float]:
        return top(self.scores(sets, query, positions), positions)

    def similarity(self, sets: list[np.ndarray], query: int, position: int) -> float:
        return float(self.scores(sets, query, np.array([position]))[0])

    def memory_bytes(self) -> int:
        return self.base.memory_bytes()


class SpatiallyVerifiedSystem:
    """System B's top `top_k` images by Jaccard similarity, re-ranked by RANSAC inliers.

    Retrieval and LSH are B's own, so only the verification differs. The score is the inlier count plus the
    Jaccard similarity, which is below 1 and so only breaks ties between equal counts.
    """

    name = "visual-words+ransac"

    def __init__(self, base: VisualWordSystem, descriptors: list[np.ndarray], keypoints: list[np.ndarray], top_k: int):
        self.base, self.index, self.top_k = base, base.index, top_k
        self.descriptors, self.keypoints = descriptors, keypoints

    def encode(self, features: FeatureSet) -> tuple[list[np.ndarray], FeatureSet]:
        return self.base.encode(features), features

    def keys(self, encoded: tuple[list[np.ndarray], FeatureSet]) -> np.ndarray:
        return self.base.keys(encoded[0])

    def verify(self, features: FeatureSet, query: int, position: int) -> int:
        return inliers(features.keypoints[query], self.keypoints[position],
                       features.descriptors[query], self.descriptors[position])

    def rank(self, encoded: tuple[list[np.ndarray], FeatureSet], query: int,
             positions: np.ndarray | None = None) -> tuple[int, float]:
        sets, features = encoded
        jaccard = self.base.scores(sets, query, positions)
        shortlist = np.argsort(-jaccard, kind="stable")[: self.top_k]
        shortlist_positions = shortlist if positions is None else positions[shortlist]
        scores = np.array([self.verify(features, query, int(p)) for p in shortlist_positions]) + jaccard[shortlist]
        return top(scores, shortlist_positions)

    def similarity(self, encoded: tuple[list[np.ndarray], FeatureSet], query: int, position: int) -> float:
        """RANSAC inliers between the query and the image at `position`, whether or not it was shortlisted."""
        return float(self.verify(encoded[1], query, position))

    def memory_bytes(self) -> int:
        stored = sum(d.nbytes for d in self.descriptors) + sum(k.nbytes for k in self.keypoints)
        return self.base.memory_bytes() + stored


def search(systems: list[System], index_ids: list[str], queries: FeatureSet, true_ids: list[str],
           kind: str, strength: float) -> list[dict]:
    """Rows in the shared results schema for every query, system and mode.

    `seconds` is the search time per query after feature extraction: ranking the whole index when
    exhaustive; retrieving the candidates and ranking only those with LSH.
    """
    position = {image_id: i for i, image_id in enumerate(index_ids)}
    rows = []
    for system in systems:
        encoded = system.encode(queries)
        keys = system.keys(encoded)
        for i, (query_id, true_id) in enumerate(zip(queries.ids, true_ids)):
            true_position = position[true_id] if true_id else -1

            start = time.perf_counter()
            exhaustive = (*system.rank(encoded, i), len(index_ids), time.perf_counter() - start)

            start = time.perf_counter()
            candidates = system.index.candidates(keys[i])
            if len(candidates):
                lsh = (*system.rank(encoded, i, candidates), len(candidates), time.perf_counter() - start)
            else:
                lsh = (-1, -np.inf, 0, time.perf_counter() - start)

            true_is_candidate = bool(true_position >= 0 and np.isin(true_position, candidates))
            true_similarity = system.similarity(encoded, i, true_position) if true_position >= 0 else np.nan
            for mode, (predicted, score, searched, seconds) in (("exhaustive", exhaustive), ("lsh", lsh)):
                rows.append(dict(
                    system=system.name, mode=mode, kind=kind, strength=strength, query_id=query_id, true_id=true_id,
                    predicted=index_ids[predicted] if predicted >= 0 else "", score=score, candidates=searched,
                    true_is_candidate=true_is_candidate, true_similarity=true_similarity, seconds=seconds,
                ))
    return rows
