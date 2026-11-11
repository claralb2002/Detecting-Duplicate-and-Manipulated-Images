from __future__ import annotations

import numpy as np
import pytest

from near_duplicates.features import FeatureSet
from near_duplicates.run import index_subset
from near_duplicates.systems import ContainmentSystem, PhashSystem, VisualWordSystem, search
from near_duplicates.visual_words import VisualVocabulary
from near_duplicates.vocabulary import incidence, word_sets


def test_word_sets_keep_alignment_around_images_without_keypoints():
    centers = np.eye(4, 128, dtype=np.float32) * 255
    descriptor = lambda word: (np.eye(4, 128)[word] * 255).astype(np.uint8)
    lists = [np.stack([descriptor(0), descriptor(0), descriptor(2)]), np.zeros((0, 128), np.uint8), np.stack([descriptor(3)])]
    sets = word_sets(lists, VisualVocabulary(centers, {}))
    assert [s.tolist() for s in sets] == [[0, 2], [], [3]]
    assert incidence(sets, 4).toarray().tolist() == [[1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1]]


def test_search_finds_planted_duplicates_and_reports_true_similarity():
    rng = np.random.default_rng(0)
    index_ids = [f"r{i}" for i in range(200)]
    centers = rng.normal(size=(256, 128)).astype(np.float32) * 30 + 120
    words = [rng.choice(256, size=30, replace=False) for _ in index_ids]
    as_descriptors = lambda ws: np.clip(np.rint(centers[ws]), 0, 255).astype(np.uint8)
    hashes = rng.integers(0, 2**64, size=len(index_ids), dtype=np.uint64)
    systems = [
        PhashSystem(hashes, bands=16, rows=8, seed=0),
        VisualWordSystem([as_descriptors(w) for w in words], VisualVocabulary(centers, {}), num_perm=128, bands=64, rows=2, seed=0),
    ]
    # Query 0 is r7 with one hash bit flipped and its word set intact; query 1 is a random distractor.
    queries = FeatureSet(["q0", "q1"], np.array([hashes[7] ^ np.uint64(1), rng.integers(0, 2**64, dtype=np.uint64)]),
                         [as_descriptors(words[7]), as_descriptors(rng.choice(256, size=30, replace=False))],
                         [np.zeros((30, 2), np.float32)] * 2)
    rows = search(systems, index_ids, queries, ["r7", ""], "jpeg", 50)
    assert len(rows) == 2 * 2 * 2
    for r in rows:
        if r["query_id"] == "q0":
            assert r["predicted"] == "r7" and r["true_is_candidate"]
            assert r["true_similarity"] == (63 / 64 if r["system"] == "phash" else 1.0)
        else:
            assert np.isnan(r["true_similarity"]) and not r["true_is_candidate"]
    exhaustive = [r for r in rows if r["mode"] == "exhaustive"]
    assert all(r["candidates"] == len(index_ids) for r in exhaustive)


def test_scoring_a_subset_equals_the_full_scores_at_those_positions():
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(128, 128)).astype(np.float32) * 30 + 120
    descriptors = [np.clip(np.rint(centers[rng.choice(128, size=20)]), 0, 255).astype(np.uint8) for _ in range(300)]
    hashes = rng.integers(0, 2**64, size=300, dtype=np.uint64)
    queries = FeatureSet(["q"], hashes[:1] ^ np.uint64(5), descriptors[:1], [np.zeros((20, 2), np.float32)])
    positions = np.array([0, 17, 42, 299])
    for system in (PhashSystem(hashes, 16, 8, seed=0), VisualWordSystem(descriptors, VisualVocabulary(centers, {}), 64, 32, 2, seed=0)):
        encoded = system.encode(queries)
        full = system.scores(encoded, 0)
        assert np.allclose(system.scores(encoded, 0, positions), full[positions])
        best = positions[np.argmax(full[positions])]
        assert system.rank(encoded, 0, positions) == (best, full[best])  # positions map back to index positions
        assert system.memory_bytes() > system.index.nbytes


def test_index_subset_keeps_required_images_and_is_reproducible():
    ids = [f"R{i:06d}" for i in range(1000)]
    required = {"R000003", "R000500", "R000999"}
    positions = index_subset(ids, required, 100, seed=0)
    assert len(positions) == len(set(positions.tolist())) == 100
    assert required <= {ids[i] for i in positions}
    assert (positions == index_subset(ids, required, 100, seed=0)).all()
    assert len(index_subset(ids, required, None, seed=0)) == 1000


def test_containment_scores_a_cropped_word_set_as_fully_contained_while_jaccard_falls():
    rng = np.random.default_rng(3)
    centers = rng.normal(size=(128, 128)).astype(np.float32) * 30 + 120
    as_descriptors = lambda ws: np.clip(np.rint(centers[ws]), 0, 255).astype(np.uint8)
    index_words = [rng.choice(128, size=40, replace=False) for _ in range(30)]
    vocabulary = VisualVocabulary(centers, {})
    words = VisualWordSystem([as_descriptors(w) for w in index_words], vocabulary, num_perm=64, bands=32, rows=2, seed=0)
    containment = ContainmentSystem(words)
    crop = word_sets([as_descriptors(index_words[5][:10])], vocabulary)  # a quarter of image 5's words
    assert containment.similarity(crop, 0, 5) == 1.0
    assert words.similarity(crop, 0, 5) == 0.25
    assert containment.rank(crop, 0)[0] == 5
    other = int(np.setdiff1d(np.arange(30), [5])[0])
    shared = len(np.intersect1d(index_words[other], index_words[5][:10]))
    assert containment.scores(crop, 0)[other] == pytest.approx(shared / 10)


def test_vocabulary_pool_holds_no_evaluation_source():
    from near_duplicates.datasets import dev_ground_truth, fit_image_ids, load_split

    for dataset in ("caltech101", "disc21"):
        try:
            pool = set(fit_image_ids(dataset))
            evaluation = set(load_split(dataset, "val_targets")) | set(load_split(dataset, "test_targets"))
            if dataset == "disc21":
                evaluation |= set(dev_ground_truth().values())
        except FileNotFoundError:
            continue
        assert pool and not pool & evaluation and pool <= set(load_split(dataset, "index"))
