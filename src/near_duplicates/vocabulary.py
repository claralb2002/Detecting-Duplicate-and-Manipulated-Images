"""Visual-word sets from the shared vocabulary (`visual_words`), fitted once per dataset, size and seed."""

from __future__ import annotations

import sys

import numpy as np
import scipy.sparse as sp

from near_duplicates.datasets import fit_image_ids
from near_duplicates.features import cache_dir
from near_duplicates.visual_words import VisualVocabulary, fit_vocabulary, sample_descriptors

KMEANS_SAMPLE = 300_000
DIAGNOSTIC_SAMPLE = 200_000


def fitted_vocabulary(dataset: str, index_ids: list[str], index_descriptors: list[np.ndarray], k: int, seed: int) -> VisualVocabulary:
    """The cached k-word vocabulary fitted on `datasets.fit_image_ids` only, so no evaluation source shapes the words."""
    pool = fit_image_ids(dataset)
    path = cache_dir(dataset) / f"vocabulary_fit{len(pool)}_k{k}_s{seed}.npz"
    if path.exists():
        return VisualVocabulary.load(path)
    position = {image_id: i for i, image_id in enumerate(index_ids)}
    descriptors = lambda image_id: {"descriptors": index_descriptors[position[image_id]]}
    sample = sample_descriptors(pool, descriptors, maximum=KMEANS_SAMPLE, seed=seed)
    print(f"k-means: {k} words from {len(sample):,} descriptors of {len(pool):,} images", file=sys.stderr)
    vocabulary = fit_vocabulary(sample, k, seed=seed, training_identity=f"{dataset} fit_image_ids {len(pool)}",
                                diagnostic_size=DIAGNOSTIC_SAMPLE)
    vocabulary.save(path)
    return vocabulary


def word_sets(descriptor_lists: list[np.ndarray], vocabulary: VisualVocabulary) -> list[np.ndarray]:
    sizes = [len(d) for d in descriptor_lists]
    words = vocabulary.assign(np.concatenate(descriptor_lists)) if sum(sizes) else np.empty(0, np.int32)
    return [np.unique(w) for w in np.split(words, np.cumsum(sizes)[:-1])]


def incidence(sets: list[np.ndarray], vocabulary: int) -> sp.csr_matrix:
    """Binary image-by-word matrix, so set intersections against the whole index are one sparse product."""
    rows = np.repeat(np.arange(len(sets)), [len(s) for s in sets])
    columns = np.concatenate(sets) if sets else np.empty(0, dtype=np.int64)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, columns)), shape=(len(sets), vocabulary))
