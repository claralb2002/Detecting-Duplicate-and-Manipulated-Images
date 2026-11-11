"""Mathematical reference, retrieval and calibration checks; no dataset required."""

from math import comb
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from near_duplicates.cache import FeatureCache
from near_duplicates.lsh import BitSamplingLSH, Match, collision_probability, verify_candidates
from near_duplicates.phash import hash_image, hash_split, hamming_distance, hamming_distances, phash
from near_duplicates.system_a import ValidationQueries, make_validation_queries, tune_lsh


class PerceptualHashTests(unittest.TestCase):
    def test_matches_explicit_cosine_dct_and_bit_order(self):
        # Independent O(N^3) definition of unnormalised DCT-II, not another FFT.
        n = np.arange(32)
        cosine = 2 * np.cos(np.pi * np.arange(32)[:, None] * (2 * n + 1) / 64)
        rng = np.random.default_rng(2807)
        for _ in range(8):
            pixels = rng.integers(0, 256, (32, 32), dtype=np.uint8)
            low = (cosine @ pixels @ cosine.T)[:8, :8]
            expected = int("".join(str(int(bit)) for bit in (low > np.median(low)).ravel()), 2)
            actual = phash(pixels)
            self.assertEqual(int(actual), expected)
            self.assertEqual(actual.dtype, np.dtype("uint64"))

    def test_uniform_images_and_strict_median_ties(self):
        self.assertEqual(phash(np.zeros((32, 32), dtype=np.uint8)), 0)
        # Only the DC coefficient exceeds zero; it occupies the highest bit.
        self.assertEqual(int(phash(np.full((32, 32), 128, dtype=np.uint8))), 1 << 63)
        self.assertEqual(int(phash(np.full((32, 32), 255, dtype=np.uint8))), 1 << 63)

    def test_requires_shared_preprocessing(self):
        for pixels in (np.zeros((16, 16), dtype=np.uint8), np.zeros((32, 32)),
                       np.zeros((32, 32, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                phash(pixels)

    def test_file_hash_cache_and_content_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "source.png"
            Image.fromarray(np.random.default_rng(7).integers(0, 256, (80, 60), dtype=np.uint8)).save(path)
            cache = FeatureCache(root / "cache")
            with patch("near_duplicates.cache.image_path", return_value=path):
                expected = hash_image(path)
                np.testing.assert_array_equal(hash_split("caltech101", ["id"], cache), [expected])
                with patch("near_duplicates.cache.load_image", side_effect=AssertionError("cache missed")):
                    self.assertEqual(int(hash_split("caltech101", ["id"], cache)[0]), int(expected))
                Image.new("L", (80, 60), 0).save(path)
                self.assertEqual(int(hash_split("caltech101", ["id"], cache)[0]), 0)
                self.assertEqual(hash_split("caltech101", [], cache).shape, (0,))

    def test_hamming_reference_boundaries_and_invalid_inputs(self):
        rng = np.random.default_rng(42)
        hashes = rng.integers(0, np.iinfo(np.uint64).max, 100, dtype=np.uint64)
        query = int(hashes[0])
        expected = [sum(a != b for a, b in zip(f"{query:064b}", f"{int(h):064b}")) for h in hashes]
        np.testing.assert_array_equal(hamming_distances(query, hashes), expected)
        self.assertEqual(hamming_distance(0, (1 << 64) - 1), 64)
        self.assertEqual(hamming_distance(query, query), 0)
        self.assertEqual(hamming_distances(0, []).size, 0)
        for bad in (-1, 1 << 64, 0.5, True, "42"):
            with self.assertRaises(ValueError):
                hamming_distance(bad, 0)
        with self.assertRaises(ValueError):
            hamming_distances(0, [[1, 2]])


class BitSamplingTests(unittest.TestCase):
    def test_seeded_tables_and_exact_hash_retrieval(self):
        hashes = np.array([0, 123, (1 << 64) - 1], dtype=np.uint64)
        first = BitSamplingLSH(hashes, ["a", "b", "c"], b=20, r=12)
        second = BitSamplingLSH(hashes, ["a", "b", "c"], b=20, r=12)
        np.testing.assert_array_equal(first.positions, second.positions)
        self.assertTrue(all(len(set(row)) == 12 for row in first.positions))
        for row, query in enumerate(hashes):
            self.assertIn(row, first.candidates(query))
        # Catalogue ownership prevents caller mutation from breaking buckets.
        hashes[:] = 0
        self.assertEqual(first.query(123, max_distance=0), [Match("b", 0)])

    def test_bucket_collision_is_verified_with_all_bits(self):
        prototype = BitSamplingLSH([], [], b=1, r=1)
        unsampled = ((1 << 64) - 1) ^ int(prototype.masks[0])
        index = BitSamplingLSH([0, unsampled], ["exact", "false_collision"], b=1, r=1)
        np.testing.assert_array_equal(index.candidates(0), [0, 1])
        self.assertEqual(index.query(0, max_distance=0, top_k=None), [Match("exact", 0)])
        self.assertEqual(index.query(0, max_distance=63, top_k=None), [Match("exact", 0), Match("false_collision", 63)])

    def test_empty_buckets_empty_catalogue_duplicates_and_ties(self):
        self.assertEqual(BitSamplingLSH([], [], b=2, r=4).query(0, max_distance=64), [])
        index = BitSamplingLSH([1, 1], ["z", "a"], b=3, r=64)
        self.assertEqual(index.query(0, max_distance=64), [])
        self.assertEqual(index.query(1, max_distance=0, top_k=None), [Match("a", 0), Match("z", 0)])
        result = verify_candidates(1, [1, 0, 1], ["z", "m", "a"], [2, 0, 0, 1], max_distance=1, top_k=None)
        self.assertEqual(result, [Match("a", 0), Match("z", 0), Match("m", 1)])

    def test_query_matches_exhaustive_verification_of_candidate_union(self):
        rng = np.random.default_rng(5)
        hashes = rng.integers(0, np.iinfo(np.uint64).max, 120, dtype=np.uint64)
        ids = [str(i) for i in range(len(hashes))]
        index = BitSamplingLSH(hashes, ids, b=6, r=4)
        for query in hashes[:20]:
            # Independently check sampled-position equality, without mask keys.
            candidates = []
            for row, value in enumerate(hashes):
                if any(all((int(query) >> int(p)) & 1 == (int(value) >> int(p)) & 1
                           for p in positions) for positions in index.positions):
                    candidates.append(row)
            np.testing.assert_array_equal(index.candidates(query), candidates)
            expected = sorted(
                [Match(ids[row], hamming_distance(query, hashes[row])) for row in candidates
                 if hamming_distance(query, hashes[row]) <= 30],
                key=lambda match: (match.distance, match.image_id),
            )[:5]
            self.assertEqual(index.query(query, max_distance=30, top_k=5), expected)

    def test_exact_collision_law_and_empirical_probability(self):
        self.assertEqual(collision_probability(0, 8, 12), 1)
        self.assertEqual(collision_probability(64, 8, 12), 0)
        self.assertEqual(collision_probability(1, 8, 64), 0)
        p = comb(52, 6) / comb(64, 6)
        self.assertAlmostEqual(collision_probability(12, 3, 6), 1 - (1 - p) ** 3)
        # A fixed pair, many independently seeded indexes. Tests the sampler too.
        hits = sum(bool(len(BitSamplingLSH([0], ["source"], b=3, r=6, seed=seed).candidates((1 << 12) - 1)))
                   for seed in range(1200))
        self.assertAlmostEqual(hits / 1200, collision_probability(12, 3, 6), delta=0.045)

    def test_invalid_configuration_and_candidate_rows(self):
        for kwargs in ({"b": 0}, {"r": 65}, {"r": 0}, {"seed": -1}, {"b": True}):
            with self.assertRaises(ValueError):
                BitSamplingLSH([0], ["id"], **kwargs)
        for hashes, ids in (([0], []), ([0, 1], ["id", "id"]), ([1.5], ["id"])):
            with self.assertRaises(ValueError):
                BitSamplingLSH(hashes, ids)
        for kwargs in ({"max_distance": -1}, {"max_distance": 65}, {"max_distance": 0, "top_k": 0}):
            with self.assertRaises(ValueError):
                BitSamplingLSH([0], ["id"]).query(0, **kwargs)
        for rows in ([-1], [1], [0.5]):
            with self.assertRaises(ValueError):
                verify_candidates(0, [0], ["id"], rows, max_distance=0)


class CalibrationTests(unittest.TestCase):
    def test_known_grid_selection_and_distractor_rejection(self):
        queries = ValidationQueries(np.array([0, (1 << 64) - 1, 1234], dtype=np.uint64),
                                    ("a", "b", None), ("probe", "probe", "probe"))
        report = tune_lsh([0, (1 << 64) - 1], ["a", "b"], queries,
                          b_values=(1, 2), r_values=(64,), thresholds=(0, 64))
        selected = report["selected"]
        self.assertEqual((selected["b"], selected["r"], selected["max_distance"]), (1, 64, 0))
        self.assertEqual(selected["recall_at_1"], 1)
        self.assertEqual(selected["precision"], 1)
        self.assertEqual(selected["distractor_false_positive_rate"], 0)
        self.assertEqual(len(report["sweep"]), 4)
        self.assertEqual(report["by_transform"]["probe"]["true_positives"], 2)

    def test_wrong_source_counts_as_false_positive_and_false_negative(self):
        queries = ValidationQueries(np.array([0, 1], dtype=np.uint64), ("z", None), ("probe", "probe"))
        report = tune_lsh([0, 0], ["z", "a"], queries, b_values=(1,), r_values=(64,), thresholds=(0,))
        self.assertEqual(report["selected"]["false_positives"], 1)
        self.assertEqual(report["selected"]["false_negatives"], 1)
        self.assertEqual(report["selected"]["recall_at_1"], 0)

    def test_generator_loads_only_validation_sources_and_preprocesses_transforms(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            Image.fromarray(np.random.default_rng(1).integers(0, 256, (80, 60), dtype=np.uint8)).save(path)
            seen = []

            def split(dataset, name):
                seen.append(name)
                return ["source"] if name == "val_targets" else ["outside"]

            with patch("near_duplicates.system_a.load_split", side_effect=split), \
                 patch("near_duplicates.system_a.image_path", return_value=path):
                queries = make_validation_queries("caltech101")
            self.assertEqual(seen, ["val_targets", "val_distractors"])
            self.assertEqual(queries.hashes.dtype, np.dtype("uint64"))
            self.assertEqual(queries.expected_ids, ("source",) * 5 + (None,) * 5)

    def test_tuning_rejects_missing_sources_and_unbalanced_queries(self):
        for expected in (("missing", None), ("a", "a")):
            queries = ValidationQueries(np.array([0, 1], dtype=np.uint64), expected, ("probe", "probe"))
            with self.assertRaises(ValueError):
                tune_lsh([0], ["a"], queries)


if __name__ == "__main__":
    unittest.main()
