"""Mathematical references for pHash and Hamming distances; no dataset required."""

import unittest

import numpy as np

from near_duplicates.phash import hamming_distance, hamming_distances, phash


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




if __name__ == "__main__":
    unittest.main()
