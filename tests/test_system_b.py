"""System B checks: real SIFT, vocabularies, minwise mathematics, banding and robust geometry."""

import itertools
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from near_duplicates.local_features import SIFTConfig, extract_local_features
from near_duplicates.lsh import BandIndex, band_keys, s_curve
from near_duplicates.minhash import EMPTY, MinHasher
from near_duplicates.phash import phash
from near_duplicates.preprocessing import PHASH_PREPROCESS, SIFT_PREPROCESS, preprocess_image
from near_duplicates.spatial import SpatialConfig, accepted_inliers, verify_spatial
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
    def test_minwise_identity_by_enumerating_every_permutation(self):
        # Exact combinatorial reference, not a probabilistic tolerance test.
        a, b = [0, 1], [1, 2]
        collisions = 0
        permutations = list(itertools.permutations(range(4)))
        for ranks in permutations:
            collisions += min(ranks[i] for i in a) == min(ranks[i] for i in b)
        self.assertEqual(collisions / len(permutations), 1/3)  # Jaccard of {0, 1} and {1, 2}
        hasher = MinHasher(4, num_perm=2000, seed=1)
        self.assertAlmostEqual(np.mean(hasher.signature(a) == hasher.signature(b)), 1/3, delta=0.04)
        np.testing.assert_array_equal(hasher.signature([0, 1, 1]), hasher.ranks[:, [0, 1]].min(axis=1))
        np.testing.assert_array_equal(hasher.ranks, MinHasher(4, num_perm=2000, seed=1).ranks)
        self.assertEqual(hasher.transform([]).shape, (0, 2000))

    def test_banding_candidate_union_and_empty_signatures(self):
        signatures = np.array([[1, 2, 3, 4], [1, 2, 9, 9], [8, 8, 3, 4], [7, 7, 7, 7]], np.uint32)
        index = BandIndex(band_keys(signatures, 2, 2))
        np.testing.assert_array_equal(index.candidates(band_keys(signatures[:1], 2, 2)[0]), [0, 1, 2])
        np.testing.assert_array_equal(MinHasher(4, num_perm=3).signature([]), [EMPTY] * 3)
        with self.assertRaises(ValueError):
            band_keys(signatures, 3, 2)
        with self.assertRaises(ValueError):
            MinHasher(4).signature([4])

    def test_banding_s_curve_matches_repeated_independent_constructions(self):
        hits = 0
        for seed in range(500):
            hasher = MinHasher(8, num_perm=8, seed=seed)
            keys = band_keys(hasher.transform([[0, 1, 2], [1, 2, 3]]), 4, 2)
            hits += len(BandIndex(keys[:1]).candidates(keys[1])) > 0
        self.assertAlmostEqual(hits / 500, s_curve(0.5, 4, 2), delta=0.06)
        self.assertEqual(s_curve(0, 4, 2), 0)
        self.assertEqual(s_curve(1, 4, 2), 1)


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
        self.assertEqual(accepted_inliers(first["keypoints"], blank["keypoints"], first["descriptors"], blank["descriptors"]), 0)
        self.assertEqual(accepted_inliers(first["keypoints"], second["keypoints"], first["descriptors"], second["descriptors"]),
                         verify_spatial(first, second).inliers)

    def test_repeated_keypoint_orientations_do_not_get_extra_geometric_votes(self):
        points = np.repeat(np.array([[0, 0], [50, 0], [0, 50]], np.float32), 2, axis=0)
        descriptors = np.random.default_rng(4).normal(size=(6, 128)).astype(np.float32)
        features = {"keypoints": points, "descriptors": descriptors}
        result = verify_spatial(features, features, SpatialConfig(min_inliers=4))
        self.assertFalse(result.accepted)
        self.assertEqual(result.matches, 3)

if __name__ == "__main__":
    unittest.main()
