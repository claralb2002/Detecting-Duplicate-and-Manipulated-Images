"""k-means visual vocabulary that turns an image's SIFT descriptors into a set of visual words."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score

from near_duplicates.datasets import DATA_DIR, fit_image_ids

KMEANS_SAMPLE = 300_000
ASSIGN_DISTANCES = 2**26  # distances per batch (256 MB of float32), whatever the vocabulary size


def descriptor_sample(descriptors: list[np.ndarray], size: int, seed: int) -> np.ndarray:
    pool = np.concatenate(descriptors)
    return pool[np.random.default_rng(seed).choice(len(pool), size=min(len(pool), size), replace=False)].astype(np.float32)


def build_vocabulary(descriptors: list[np.ndarray], k: int, seed: int = 0, sample_size: int = KMEANS_SAMPLE) -> np.ndarray:
    sample = descriptor_sample(descriptors, sample_size, seed)
    print(f"k-means: {k} words from {len(sample):,} descriptors", file=sys.stderr)
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10_000, n_init=3, random_state=seed).fit(sample)
    return kmeans.cluster_centers_.astype(np.float32)


def davies_bouldin(sample: np.ndarray, centers: np.ndarray) -> float:
    """Davies-Bouldin index of the vocabulary on a descriptor sample labelled by nearest word; lower is better."""
    return float(davies_bouldin_score(sample, nearest_word(sample, centers)))


def cached_vocabulary(path: Path, descriptors: list[np.ndarray], k: int, seed: int) -> np.ndarray:
    if path.exists():
        return np.load(path)["centers"]
    centers = build_vocabulary(descriptors, k, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, centers=centers)
    os.replace(tmp, path)
    return centers


def fitted_vocabulary(dataset: str, index_ids: list[str], index_descriptors: list[np.ndarray], k: int, seed: int) -> np.ndarray:
    """The cached k-word vocabulary fitted on `datasets.fit_image_ids` only, so no evaluation source shapes the words."""
    pool = fit_image_ids(dataset)
    position = {image_id: i for i, image_id in enumerate(index_ids)}
    path = DATA_DIR / "cache" / dataset / f"vocabulary_fit{len(pool)}_k{k}_s{seed}.npz"
    return cached_vocabulary(path, [index_descriptors[position[i]] for i in pool], k, seed)


def nearest_word(descriptors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Index of the nearest center per descriptor; ||c||^2 - 2 d.c ranks centers like the squared distance."""
    center_norms = (centers**2).sum(axis=1)
    batch = max(1, ASSIGN_DISTANCES // len(centers))
    words = [
        np.argmin(center_norms[None, :] - 2.0 * descriptors[start : start + batch].astype(np.float32) @ centers.T, axis=1)
        for start in range(0, len(descriptors), batch)
    ]
    return np.concatenate(words) if words else np.empty(0, dtype=np.int64)


def word_sets(descriptor_lists: list[np.ndarray], centers: np.ndarray) -> list[np.ndarray]:
    sizes = [len(d) for d in descriptor_lists]
    words = nearest_word(np.concatenate(descriptor_lists), centers)
    return [np.unique(w) for w in np.split(words, np.cumsum(sizes)[:-1])]


def incidence(sets: list[np.ndarray], vocabulary: int) -> sp.csr_matrix:
    """Binary image-by-word matrix, so set intersections against the whole index are one sparse product."""
    rows = np.repeat(np.arange(len(sets)), [len(s) for s in sets])
    columns = np.concatenate(sets) if sets else np.empty(0, dtype=np.int64)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, columns)), shape=(len(sets), vocabulary))
