from __future__ import annotations

import numpy as np

from near_duplicates.features import FeatureSet
from near_duplicates.run import index_subset
from near_duplicates.systems import PhashSystem, VisualWordSystem, search
from near_duplicates.vocabulary import incidence, nearest_word, word_sets


def test_nearest_word_matches_brute_force_distance():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(64, 128)).astype(np.float32) * 40 + 100
    descriptors = rng.integers(0, 256, size=(500, 128), dtype=np.uint8)
    distances = ((descriptors[:, None, :].astype(float) - centers[None].astype(float)) ** 2).sum(axis=2)
    assert (nearest_word(descriptors, centers) == distances.argmin(axis=1)).all()


def test_word_sets_keep_alignment_around_images_without_keypoints():
    centers = np.eye(4, 128, dtype=np.float32) * 255
    descriptor = lambda word: (np.eye(4, 128)[word] * 255).astype(np.uint8)
    lists = [np.stack([descriptor(0), descriptor(0), descriptor(2)]), np.zeros((0, 128), np.uint8), np.stack([descriptor(3)])]
    sets = word_sets(lists, centers)
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
        VisualWordSystem([as_descriptors(w) for w in words], centers, num_perm=128, bands=64, rows=2, seed=0),
    ]
    # Query 0 is r7 with one hash bit flipped and its word set intact; query 1 is a random distractor.
    queries = FeatureSet(["q0", "q1"], np.array([hashes[7] ^ np.uint64(1), rng.integers(0, 2**64, dtype=np.uint64)]),
                         [as_descriptors(words[7]), as_descriptors(rng.choice(256, size=30, replace=False))])
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
    queries = FeatureSet(["q"], hashes[:1] ^ np.uint64(5), descriptors[:1])
    positions = np.array([0, 17, 42, 299])
    for system in (PhashSystem(hashes, 16, 8, seed=0), VisualWordSystem(descriptors, centers, 64, 32, 2, seed=0)):
        encoded = system.encode(queries)
        assert np.allclose(system.scores(encoded, 0, positions), system.scores(encoded, 0)[positions])
        assert system.memory_bytes() > system.index.nbytes


def test_index_subset_keeps_required_images_and_is_reproducible():
    ids = [f"R{i:06d}" for i in range(1000)]
    required = {"R000003", "R000500", "R000999"}
    positions = index_subset(ids, required, 100, seed=0)
    assert len(positions) == len(set(positions.tolist())) == 100
    assert required <= {ids[i] for i in positions}
    assert (positions == index_subset(ids, required, 100, seed=0)).all()
    assert len(index_subset(ids, required, None, seed=0)) == 1000
