"""Visual-word retrieval, exact verification and optional spatial reranking."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from near_duplicates.minhash import MinHasher, MinHashLSH, word_set
from near_duplicates.spatial import SpatialConfig, verify_spatial


@dataclass(frozen=True)
class VisualMatch:
    image_id: str
    score: float
    inliers: int = 0
    inlier_ratio: float = 0.0


@dataclass(frozen=True)
class RetrievalResult:
    matches: tuple[VisualMatch, ...]
    candidates: np.ndarray


class VisualWordIndex:
    """Descriptor-free retrieval index; load descriptor caches only for reranking."""

    def __init__(self, sets: Sequence[Sequence[int]], image_ids: Sequence[str], hasher: MinHasher, *, b: int, r: int):
        if len(sets) != len(image_ids):
            raise ValueError("provide one visual-word set per image")
        self.hasher, self.image_ids = hasher, tuple(image_ids)
        self.sets = tuple(word_set(s, hasher.vocabulary_size) for s in sets)
        self.signatures = hasher.transform(self.sets)
        self.lsh = MinHashLSH(self.signatures, image_ids, b=b, r=r)
        self.sizes = np.array([len(s) for s in self.sets], dtype=np.float32)
        offsets = np.array([0, *np.cumsum(self.sizes, dtype=np.int64)], dtype=np.int64)
        indices = np.concatenate(self.sets) if self.sets else np.empty(0, np.uint32)
        self.matrix = csr_matrix((np.ones(len(indices), dtype=np.float32), indices, offsets),
                                  shape=(len(sets), hasher.vocabulary_size))
        self._id_rank = np.empty(len(image_ids), np.intp)
        self._id_rank[np.argsort(np.array(image_ids), kind="stable")] = np.arange(len(image_ids))

    def candidates(self, words: Sequence[int]) -> np.ndarray:
        return self.lsh.candidates(self.hasher.signature(words))

    def scores(self, words: Sequence[int], *, metric: str = "jaccard", rows: np.ndarray | None = None) -> np.ndarray:
        words = word_set(words, self.hasher.vocabulary_size)
        if metric not in ("jaccard", "containment"):
            raise ValueError("metric must be jaccard or containment")
        if not len(words):
            return np.zeros(len(self.image_ids) if rows is None else len(rows), np.float32)
        vector = csr_matrix((np.ones(len(words), np.float32), (words, np.zeros(len(words), np.int32))),
                            shape=(self.hasher.vocabulary_size, 1))
        matrix = self.matrix if rows is None else self.matrix[rows]
        sizes = self.sizes if rows is None else self.sizes[rows]
        overlap = (matrix @ vector).toarray().ravel()
        denominator = len(words) + sizes - overlap if metric == "jaccard" else np.full(len(overlap), len(words))
        return np.divide(overlap, denominator, out=np.zeros_like(overlap), where=denominator > 0)

    def ranked_rows(self, words: Sequence[int], *, metric: str = "jaccard", threshold: float = 0.0,
                    exhaustive: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must lie in [0, 1]")
        candidates = np.flatnonzero(self.sizes > 0) if exhaustive else self.candidates(words)
        scores = np.zeros(len(self.image_ids), np.float32)
        scores[candidates] = self.scores(words, metric=metric, rows=candidates)
        # A zero-overlap or featureless image never produces a match, even at threshold zero.
        accepted = candidates[(scores[candidates] >= threshold) & (scores[candidates] > 0)]
        order = np.lexsort((self._id_rank[accepted], -scores[accepted]))
        return accepted[order], scores, candidates

    def query(self, words: Sequence[int], *, metric: str = "jaccard", threshold: float = 0.0,
              exhaustive: bool = False, top_k: int = 1, query_features: dict | None = None,
              reference_features: Callable[[str], dict] | None = None,
              spatial: SpatialConfig | None = None, rerank_depth: int = 10) -> RetrievalResult:
        if type(top_k) is not int or top_k < 1 or type(rerank_depth) is not int or rerank_depth < 1:
            raise ValueError("top_k and rerank_depth must be positive integers")
        rows, scores, candidates = self.ranked_rows(words, metric=metric, threshold=threshold, exhaustive=exhaustive)
        matches = []
        if spatial is None:
            matches = [VisualMatch(self.image_ids[row], float(scores[row])) for row in rows[:top_k]]
        else:
            if query_features is None or reference_features is None:
                raise ValueError("spatial verification requires query and reference features")
            for row in rows[:rerank_depth]:
                check = verify_spatial(query_features, reference_features(self.image_ids[row]), spatial)
                if check.accepted:
                    matches.append(VisualMatch(self.image_ids[row], float(scores[row]), check.inliers, check.inlier_ratio))
            matches.sort(key=lambda m: (-m.inliers, -m.inlier_ratio, -m.score, m.image_id))
        return RetrievalResult(tuple(matches[:top_k]), candidates)
