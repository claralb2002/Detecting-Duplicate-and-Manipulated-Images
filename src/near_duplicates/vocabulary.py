"""k-means visual vocabulary that turns an image's SIFT descriptors into a set of visual words."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans

KMEANS_SAMPLE = 300_000
ASSIGN_BATCH = 50_000


def build_vocabulary(descriptors: list[np.ndarray], k: int, seed: int = 0, sample_size: int = KMEANS_SAMPLE) -> np.ndarray:
    pool = np.concatenate(descriptors)
    rng = np.random.default_rng(seed)
    sample = pool[rng.choice(len(pool), size=min(len(pool), sample_size), replace=False)].astype(np.float32)
    print(f"k-means: {k} words from {len(sample):,} of {len(pool):,} descriptors", file=sys.stderr)
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10_000, n_init=3, random_state=seed).fit(sample)
    return kmeans.cluster_centers_.astype(np.float32)


def cached_vocabulary(path: Path, descriptors: list[np.ndarray], k: int, seed: int) -> np.ndarray:
    if path.exists():
        return np.load(path)["centers"]
    centers = build_vocabulary(descriptors, k, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, centers=centers)
    os.replace(tmp, path)
    return centers


def nearest_word(descriptors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Index of the nearest center per descriptor; ||c||^2 - 2 d.c ranks centers like the squared distance."""
    center_norms = (centers**2).sum(axis=1)
    words = [
        np.argmin(center_norms[None, :] - 2.0 * descriptors[start : start + ASSIGN_BATCH].astype(np.float32) @ centers.T, axis=1)
        for start in range(0, len(descriptors), ASSIGN_BATCH)
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
