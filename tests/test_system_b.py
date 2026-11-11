"""System B checks: real SIFT, set/minwise mathematics and robust geometry."""

import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from near_duplicates.cache import FeatureCache
from near_duplicates.local_features import LocalFeatureStore, SIFTConfig, extract_local_features, export_feature_bundle
from near_duplicates.local_queries import QuerySpec, prepare_validation_queries
from near_duplicates.minhash import EMPTY, MinHasher, MinHashLSH, candidate_probability, containment, estimated_jaccard, jaccard
from near_duplicates.phash import phash
from near_duplicates.preprocessing import PHASH_PREPROCESS, SIFT_PREPROCESS, load_image, preprocess_image
from near_duplicates.retrieval_metrics import retrieval_metrics, wilson_interval
from near_duplicates.spatial import SpatialConfig, SpatialResult, verify_spatial
from near_duplicates.system_b import CSV_FIELDS, select_parameters, select_spatial, sweep_words
from near_duplicates.visual_index import VisualWordIndex
from near_duplicates.visual_words import VisualVocabulary, fit_vocabulary, sample_descriptors
from PIL import Image

cv2.setNumThreads(1)


def texture():
    rng = np.random.default_rng(41)
    gray = rng.integers(0, 256, (220, 300), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.7)
    return np.repeat(gray[..., None], 3, axis=2)


class LocalFeaturesTests(unittest.TestCase):
    def test_sift_cap_order_geometry_and_joint_phash(self):
        rgb = texture()
        features = extract_local_features(rgb, SIFTConfig(max_features=30, max_side=160))
        n = len(features["descriptors"])
        self.assertGreaterEqual(n, 4)
        self.assertLessEqual(n, 30)
        self.assertEqual(features["descriptors"].shape, (n, 128))
        self.assertEqual(features["descriptors"].dtype, np.float32)
        self.assertEqual(features["keypoints"].shape, (n, 2))
        self.assertTrue(np.all(np.diff(features["responses"]) <= 0))
        self.assertLessEqual(int(features["image_shape"].max()), 160)
        self.assertEqual(int(features["hashes"][0]), int(phash(preprocess_image(Image.fromarray(rgb), PHASH_PREPROCESS))))
        np.testing.assert_array_equal(features["offsets"], [0, n])
        self.assertEqual((SIFT_PREPROCESS.mode, SIFT_PREPROCESS.max_side), ("RGB", 512))

    def test_empty_image_and_invalid_inputs(self):
        result = extract_local_features(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(result["descriptors"].shape, (0, 128))
        self.assertEqual(result["keypoints"].shape, (0, 2))
        for image in (np.zeros((8, 8)), np.zeros((8, 8, 3)), np.zeros((0, 8, 3), np.uint8)):
            with self.assertRaises(ValueError):
                extract_local_features(image)
        with self.assertRaises(ValueError):
            SIFTConfig(max_features=0)

    def test_joint_cache_decodes_once_and_invalidates_changed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.png"
            Image.fromarray(texture()).save(path)
            cache = FeatureCache(Path(folder) / "cache")
            store = LocalFeatureStore("caltech101", cache=cache)
            with patch("near_duplicates.cache.image_path", return_value=path), \
                 patch("near_duplicates.cache.load_image", wraps=load_image) as decode:
                first = store.get("id")
                second = store.get("id")
                self.assertEqual(decode.call_count, 1)
                np.testing.assert_array_equal(first["descriptors"], second["descriptors"])
                Image.new("RGB", (100, 100)).save(path)
                self.assertEqual(len(store.get("id")["descriptors"]), 0)
                self.assertEqual(decode.call_count, 2)

    def test_validation_sources_only_and_one_decode_for_all_transforms(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.png"
            Image.fromarray(texture()).save(path)
            store = LocalFeatureStore("caltech101", cache=FeatureCache(Path(folder) / "cache"))
            seen = []

            def split(dataset, name):
                seen.append(name)
                self.assertIn(name, ("val_targets", "val_distractors"))
                return [name]

            with patch("near_duplicates.local_queries.load_split", side_effect=split), \
                 patch("near_duplicates.local_queries.image_path", return_value=path), \
                 patch("near_duplicates.cache.image_path", return_value=path), \
                 patch("near_duplicates.local_queries.load_image", wraps=load_image) as decode:
                queries = prepare_validation_queries(store)
                self.assertEqual(decode.call_count, 2)
                repeat = prepare_validation_queries(store)
                self.assertEqual(decode.call_count, 2)
            self.assertEqual([q.identity for q in queries], [q.identity for q in repeat])
            self.assertEqual(len(queries), 10)
            self.assertTrue(all(q.true_id is None for q in queries[5:]))

    def test_feature_bundle_offsets_and_no_pickle(self):
        features = extract_local_features(texture(), SIFTConfig(max_features=12))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bundle.npz"
            export_feature_bundle(path, ["a", "b"], lambda image_id: features)
            with np.load(path, allow_pickle=False) as stored:
                np.testing.assert_array_equal(stored["offsets"], [0, 12, 24])
                self.assertEqual(stored["descriptors"].shape, (24, 128))
                self.assertEqual(stored["hashes"].shape, (2,))


class VocabularyTests(unittest.TestCase):
    def test_assignment_matches_brute_force_and_sets_remove_counts(self):
        rng = np.random.default_rng(6)
        centers = rng.normal(size=(7, 128)).astype(np.float32)
        descriptors = rng.normal(size=(19, 128)).astype(np.float32)
        vocab = VisualVocabulary(centers, {"test": True})
        expected = np.argmin(((descriptors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        np.testing.assert_array_equal(vocab.assign(descriptors, batch_size=3), expected)
        np.testing.assert_array_equal(vocab.words(np.repeat(centers[:2], 3, axis=0)), [0, 1])
        self.assertEqual(vocab.words(np.empty((0, 128), np.float32)).size, 0)

    def test_vocabulary_fit_persistence_and_identity(self):
        rng = np.random.default_rng(8)
        points = np.concatenate([rng.normal(i * 30, 1, (40, 128)) for i in range(3)]).astype(np.float32)
        vocab = fit_vocabulary(points, 3, seed=2, max_iter=20)
        labels = vocab.assign(points)
        self.assertEqual(len(set(labels[:40])), 1)
        self.assertEqual(len(set(labels[40:80])), 1)
        self.assertEqual(len(set(labels)), 3)
        self.assertIsNotNone(vocab.metadata["davies_bouldin"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vocab.npz"
            vocab.save(path)
            loaded = VisualVocabulary.load(path)
            self.assertEqual(loaded.identity, vocab.identity)
            np.testing.assert_array_equal(loaded.assign(points), labels)
        changed = VisualVocabulary(vocab.centers + 1, vocab.metadata)
        self.assertNotEqual(changed.identity, vocab.identity)

    def test_uniform_descriptor_sampling_and_empty_pool(self):
        features = {"a": {"descriptors": np.repeat(np.arange(3, dtype=np.float32)[:, None], 128, axis=1)},
                    "b": {"descriptors": np.repeat(np.arange(3, 10, dtype=np.float32)[:, None], 128, axis=1)}}
        sampled = sample_descriptors(["a", "b"], features.__getitem__, maximum=4, seed=22)
        chosen = np.sort(np.random.default_rng(22).choice(10, 4, replace=False))
        np.testing.assert_array_equal(sampled[:, 0], chosen)
        with self.assertRaises(ValueError):
            sample_descriptors([], features.__getitem__)


class MinHashTests(unittest.TestCase):
    def test_set_similarities_empty_and_directional_containment(self):
        self.assertAlmostEqual(jaccard([1, 1, 2], [2, 3]), 1/3)
        self.assertEqual(containment([1, 2], [0, 1, 2, 3]), 1)
        self.assertEqual(containment([0, 1, 2, 3], [1, 2]), 0.5)
        self.assertEqual(jaccard([], []), 0)
        self.assertEqual(containment([], [1]), 0)

    def test_minwise_identity_by_enumerating_every_permutation(self):
        # Exact combinatorial reference, not a probabilistic tolerance test.
        a, b = [0, 1], [1, 2]
        collisions = 0
        permutations = list(itertools.permutations(range(4)))
        for ranks in permutations:
            collisions += min(ranks[i] for i in a) == min(ranks[i] for i in b)
        self.assertEqual(collisions / len(permutations), jaccard(a, b))
        hasher = MinHasher(4, num_perm=2000, seed=1)
        self.assertAlmostEqual(estimated_jaccard(hasher.signature(a), hasher.signature(b)), 1/3, delta=0.04)
        np.testing.assert_array_equal(hasher.signature([0, 1, 1]), hasher.ranks[:, [0, 1]].min(axis=1))
        np.testing.assert_array_equal(hasher.ranks, MinHasher(4, num_perm=2000, seed=1).ranks)
        self.assertEqual(hasher.transform([]).shape, (0, 2000))

    def test_banding_candidate_union_and_empty_signatures(self):
        signatures = np.array([[1, 2, 3, 4], [1, 2, 9, 9], [8, 8, 3, 4], [7, 7, 7, 7], [EMPTY]*4], np.uint32)
        lsh = MinHashLSH(signatures, ["a", "b", "c", "d", "empty"], b=2, r=2)
        np.testing.assert_array_equal(lsh.candidates(signatures[0]), [0, 1, 2])
        self.assertEqual(lsh.candidates(np.full(4, EMPTY, np.uint32)).size, 0)
        self.assertEqual(estimated_jaccard(signatures[-1], signatures[-1]), 0)
        with self.assertRaises(ValueError):
            MinHashLSH(signatures, ["a", "b", "c", "d", "e"], b=3, r=2)
        with self.assertRaises(ValueError):
            MinHasher(4).signature([4])

    def test_banding_s_curve_matches_repeated_independent_constructions(self):
        hits = 0
        for seed in range(500):
            hasher = MinHasher(8, num_perm=8, seed=seed)
            signatures = hasher.transform([[0, 1, 2], [1, 2, 3]])
            index = MinHashLSH(signatures[:1], ["a"], b=4, r=2)
            hits += len(index.candidates(signatures[1])) > 0
        self.assertAlmostEqual(hits / 500, candidate_probability(0.5, 4, 2), delta=0.06)
        self.assertEqual(candidate_probability(0, 4, 2), 0)
        self.assertEqual(candidate_probability(1, 4, 2), 1)


class GeometryAndRetrievalTests(unittest.TestCase):
    def test_ransac_recovers_homography_despite_outliers(self):
        rng = np.random.default_rng(2)
        points = rng.uniform(0, 200, (30, 2)).astype(np.float32)
        desc = rng.normal(size=(30, 128)).astype(np.float32)
        target = points * np.array([0.8, 0.9]) + [12, 18]
        target[-7:] = rng.uniform(-300, 300, (7, 2))
        query = {"descriptors": desc, "keypoints": points}
        reference = {"descriptors": desc.copy(), "keypoints": target.astype(np.float32)}
        result = verify_spatial(query, reference)
        self.assertTrue(result.accepted)
        self.assertGreaterEqual(result.inliers, 23)
        self.assertLess(result.inliers, 30)
        projected = cv2.perspectiveTransform(points[:20, None, :], result.homography).reshape(-1, 2)
        np.testing.assert_allclose(projected, target[:20], atol=0.05)
        line = np.column_stack([np.arange(30), np.zeros(30)]).astype(np.float32)
        self.assertFalse(verify_spatial({**query, "keypoints": line}, {**reference, "keypoints": line}).accepted)

    def test_real_sift_geometric_verification_and_blank_rejection(self):
        rgb = texture()
        shifted = cv2.warpAffine(rgb, np.float32([[1, 0, 8], [0, 1, 12]]), (300, 220))
        first, second = extract_local_features(rgb), extract_local_features(shifted)
        self.assertTrue(verify_spatial(first, second).accepted)
        blank = extract_local_features(np.zeros_like(rgb))
        self.assertFalse(verify_spatial(first, blank).accepted)

    def test_repeated_keypoint_orientations_do_not_get_extra_geometric_votes(self):
        points = np.repeat(np.array([[0, 0], [50, 0], [0, 50]], np.float32), 2, axis=0)
        descriptors = np.random.default_rng(4).normal(size=(6, 128)).astype(np.float32)
        features = {"keypoints": points, "descriptors": descriptors}
        result = verify_spatial(features, features, SpatialConfig(min_inliers=4))
        self.assertFalse(result.accepted)
        self.assertEqual(result.matches, 3)

    def test_index_scores_verification_empty_and_deterministic_ties(self):
        sets, ids = [[0, 1], [0, 1], [0, 1, 2, 3], []], ["z", "a", "large", "blank"]
        index = VisualWordIndex(sets, ids, MinHasher(5), b=64, r=2)
        np.testing.assert_allclose(index.scores([0, 1]), [1, 1, .5, 0])
        np.testing.assert_allclose(index.scores([0, 1], metric="containment"), [1, 1, 1, 0])
        result = index.query([0, 1], threshold=0.9, top_k=3)
        self.assertEqual([m.image_id for m in result.matches], ["a", "z"])
        self.assertEqual(index.query([], exhaustive=True).matches, ())
        self.assertEqual(index.query([4], exhaustive=True).matches, ())

    def test_spatial_reranking_rejects_visual_word_collision(self):
        good = extract_local_features(texture())
        bad = extract_local_features(np.zeros((220, 300, 3), np.uint8))
        index = VisualWordIndex([[0, 1], [0, 1]], ["a_bad", "z_good"], MinHasher(4), b=64, r=2)
        self.assertEqual(index.query([0, 1]).matches[0].image_id, "a_bad")
        result = index.query([0, 1], query_features=good, reference_features={"a_bad": bad, "z_good": good}.__getitem__,
                             spatial=SpatialConfig(), rerank_depth=2)
        self.assertEqual(result.matches[0].image_id, "z_good")
        self.assertGreaterEqual(result.matches[0].inliers, 6)

    def test_sweep_known_answers_and_cost_aware_selection(self):
        queries = [QuerySpec("a", "a", "probe", 1, 1), QuerySpec("outside", None, "probe", 1, 1)]
        rows, curves = sweep_words([[0, 1], [2, 3]], [[0, 1], [4]], ["a", "b"], queries,
                                   MinHasher(5, num_perm=4), bands=((2, 2),), thresholds=(0.1, 0.5))
        self.assertTrue(all(row["f1"] == 1 for row in rows))
        self.assertTrue(curves)
        base = {"vocabulary_size": 4, "b": 2, "r": 2, "metric": "jaccard", "threshold": .5,
                "recall_at_1": .9, "precision": .9, "seed": 1}
        sweep = [{**base, "f1": .9, "mean_candidates": 100},
                 {**base, "b": 1, "r": 4, "f1": .897, "mean_candidates": 10}]
        self.assertEqual(select_parameters(sweep)["mean_candidates"], 10)
        self.assertEqual(select_parameters(sweep, f1_tolerance=0)["mean_candidates"], 100)
        self.assertEqual(CSV_FIELDS[:11], ("system", "mode", "kind", "strength", "true_id", "predicted", "score",
                                         "candidates", "true_is_candidate", "true_similarity", "lsh_seconds"))

    def test_wrong_source_counts_as_both_error_types_and_wilson_interval(self):
        metrics = retrieval_metrics(np.array([1, -1, 0]), np.array([0, 1, -1]))
        self.assertEqual((metrics["false_positives"], metrics["false_negatives"]), (2, 2))
        low, high = wilson_interval(250, 500)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)
        self.assertIsNone(wilson_interval(0, 0))

    def test_spatial_calibration_can_accept_low_word_overlap(self):
        valid = SpatialResult(True, 12, 12, 1.0, np.eye(3))
        accidental = SpatialResult(True, 12, 4, 1/3, np.eye(3))
        records = [{"query": 0, "mode": "lsh", "checks": [(0, 0.1, valid)]},
                   {"query": 1, "mode": "lsh", "checks": [(0, 0.3, accidental)]}]
        selected, sweep = select_spatial({2807: (records, np.array([0, -1]))})
        self.assertEqual(selected["mean_f1"], 1.0)
        self.assertLessEqual(selected["threshold"], 0.1)
        self.assertGreater(selected["min_inliers"], 4)
        self.assertEqual(len(sweep), 36)


if __name__ == "__main__":
    unittest.main()
