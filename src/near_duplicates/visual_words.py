"""Visual vocabularies fitted only on the committed fitting pool (Week 6)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score
from threadpoolctl import threadpool_limits

from near_duplicates.local_features import save_arrays


def descriptor_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 128 or not np.isfinite(values).all():
        raise ValueError("descriptors must be a finite N x 128 array")
    return np.ascontiguousarray(values)


def sample_descriptors(
    image_ids: Sequence[str], get_features: Callable[[str], dict], *, maximum: int = 2_000_000, seed: int = 2807,
) -> np.ndarray:
    """Uniform sample without replacement from all descriptors, using two passes.

    Counts are read first; the second pass copies only sampled global row IDs.
    This avoids concatenating the whole descriptor corpus in memory. Call with
    fit_image_ids(dataset), never the complete index or validation queries.
    """
    if type(maximum) is not int or maximum < 1:
        raise ValueError("maximum must be positive")
    counts = [len(get_features(i)["descriptors"]) for i in image_ids]
    total = sum(counts)
    if not total:
        raise ValueError("fitting pool has no SIFT descriptors")
    chosen = np.sort(np.random.default_rng(seed).choice(total, size=min(total, maximum), replace=False))
    sample = np.empty((len(chosen), 128), np.float32)
    offset = 0
    for image_id, count in zip(image_ids, counts):
        left, right = np.searchsorted(chosen, [offset, offset + count])
        if right > left:
            sample[left:right] = get_features(image_id)["descriptors"][chosen[left:right] - offset]
        offset += count
    return sample


@dataclass
class VisualVocabulary:
    centers: np.ndarray
    metadata: dict

    def __post_init__(self):
        if not isinstance(self.metadata, dict):
            raise ValueError("vocabulary metadata must be a dictionary")
        self.centers = descriptor_array(self.centers).copy()
        if not len(self.centers):
            raise ValueError("vocabulary must contain at least one center")
        self.centers.flags.writeable = False

    @property
    def identity(self) -> str:
        digest = hashlib.sha256(self.centers.astype("<f4").tobytes())
        digest.update(json.dumps(self.metadata, sort_keys=True, allow_nan=False).encode())
        return digest.hexdigest()

    def assign(self, descriptors: np.ndarray, *, batch_size: int = 8192) -> np.ndarray:
        """Exact nearest Euclidean center; bounded batch x vocabulary workspace.

        Each batch is converted to float32 on its own, so millions of uint8 descriptors from a feature cache
        never need a float32 copy, and the product uses every BLAS thread: the argmin does not depend on them.
        """
        descriptors = np.asarray(descriptors)
        if descriptors.ndim != 2 or descriptors.shape[1] != 128:
            raise ValueError("descriptors must be an N x 128 array")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        result = np.empty(len(descriptors), np.int32)
        norms = np.einsum("ij,ij->i", self.centers, self.centers)
        for start in range(0, len(descriptors), batch_size):
            block = descriptor_array(descriptors[start:start + batch_size])
            # Query squared norms are constant across centers and cancel in argmin.
            distances = norms[None, :] - 2 * block @ self.centers.T
            result[start:start + len(block)] = np.argmin(distances, axis=1)
        return result

    def words(self, descriptors: np.ndarray) -> np.ndarray:
        """Binary visual-word set: descriptor multiplicity is discarded."""
        return np.unique(self.assign(descriptors))

    def save(self, path: str | Path) -> None:
        save_arrays(path, self.metadata, centers=self.centers)

    @classmethod
    def load(cls, path: str | Path) -> "VisualVocabulary":
        with np.load(path, allow_pickle=False) as stored:
            return cls(stored["centers"], json.loads(stored["metadata"].item()))


def fit_vocabulary(descriptors: np.ndarray, size: int, *, seed: int = 2807, training_identity: str = "",
                   batch_size: int = 2048, max_iter: int = 100, diagnostic_size: int = 5000) -> VisualVocabulary:
    descriptors = descriptor_array(descriptors)
    if type(size) is not int or not 2 <= size <= len(descriptors):
        raise ValueError("vocabulary size must be between 2 and the number of sampled descriptors")
    model = MiniBatchKMeans(n_clusters=size, batch_size=max(batch_size, size), random_state=seed,
                            max_iter=max_iter, n_init=3, max_no_improvement=10, reassignment_ratio=0.01)
    with threadpool_limits(limits=1):
        model.fit(descriptors)
    # Internal compactness diagnostic on a seeded bounded sample, not retrieval quality; with many words the
    # sample must hold many descriptors per word, or most clusters are singletons and the index is meaningless.
    rows = np.random.default_rng(seed).choice(len(descriptors), min(diagnostic_size, len(descriptors)), replace=False)
    with threadpool_limits(limits=1):
        labels = model.predict(descriptors[rows])
    occupied = len(np.unique(labels))
    db = None
    if 1 < occupied < len(rows):
        with threadpool_limits(limits=1):
            db = float(davies_bouldin_score(descriptors[rows], labels))
    return VisualVocabulary(model.cluster_centers_, {
        "version": 1, "size": size, "seed": seed, "training_identity": training_identity,
        "sample_sha256": hashlib.sha256(descriptors.astype("<f4").tobytes()).hexdigest(),
        "training_descriptors": len(descriptors), "sklearn": sklearn.__version__,
        "algorithm": "MiniBatchKMeans", "batch_size": max(batch_size, size), "max_iter": max_iter,
        "n_init": 3, "inertia_per_descriptor": float(model.inertia_ / len(descriptors)),
        "davies_bouldin": db, "diagnostic_sample": len(rows), "diagnostic_occupied_clusters": occupied,
    })
