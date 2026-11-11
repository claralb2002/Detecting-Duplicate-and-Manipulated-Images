"""Synthetic checks for preprocessing, the dataset audit and split isolation."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from near_duplicates.datasets import dev_ground_truth, fit_image_ids
from near_duplicates.preprocessing import PHASH_PREPROCESS, PreprocessConfig, load_image
from near_duplicates.validate import validate_dataset


class SetupDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.png"
        Image.new("RGB", (80, 40), (200, 40, 20)).save(self.source)

    def test_resize_modes_and_no_upscaling(self):
        gray = load_image(self.source, PHASH_PREPROCESS)
        self.assertEqual(gray.shape, (32, 32))
        self.assertEqual(gray.dtype, np.uint8)
        self.assertEqual(load_image(self.source, PreprocessConfig(max_side=20)).shape, (10, 20))
        self.assertEqual(load_image(self.source, PreprocessConfig(mode="RGB", max_side=100)).shape, (40, 80, 3))

    def test_orientation_and_transparency(self):
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (80, 40)).save(self.source, exif=exif)
        self.assertEqual(load_image(self.source, PreprocessConfig()).shape, (80, 40))
        Image.new("RGBA", (2, 2), (0, 0, 0, 0)).save(self.source)
        self.assertTrue(np.all(load_image(self.source) == 255))

    def test_invalid_settings_and_corrupt_image(self):
        for kwargs in ({"mode": "BGR"}, {"max_side": 0}, {"square_size": 32, "max_side": 10}):
            with self.assertRaises(ValueError):
                PreprocessConfig(**kwargs)
        self.source.write_bytes(b"broken image")
        with self.assertRaisesRegex(ValueError, "source.png"):
            load_image(self.source)

    def test_fitting_pool_excludes_evaluation_sources(self):
        splits = {"index": ["a", "b", "c", "d"], "val_targets": ["b"], "test_targets": ["c"]}
        with patch("near_duplicates.datasets.load_split", side_effect=lambda d, n: splits[n]):
            self.assertEqual(fit_image_ids("caltech101"), ["a", "d"])
            with patch("near_duplicates.datasets.dev_ground_truth", return_value={"Q00000": "d"}):
                self.assertEqual(fit_image_ids("disc21"), ["a"])

    def test_ground_truth_optional_header_and_duplicates(self):
        ground_truth = self.root / "dev_ground_truth.csv"
        with patch("near_duplicates.datasets.DISC21_DIR", self.root):
            for header in ("", "query_id,reference_id\n"):
                ground_truth.write_text(header + "Q00000,R000001\nQ00001,\n")
                self.assertEqual(dev_ground_truth(), {"Q00000": "R000001"})
            ground_truth.write_text("Q00000,R000001\nQ00000,R000002\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                dev_ground_truth()
            ground_truth.write_text("Q00000,\nQ00000,R000001\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                dev_ground_truth()
            ground_truth.write_text("invalid-query,\n")
            with self.assertRaisesRegex(ValueError, "invalid query ID"):
                dev_ground_truth()
            ground_truth.write_text("Q00000,R1000000\n")
            with self.assertRaisesRegex(ValueError, "invalid reference ID"):
                dev_ground_truth()

    def test_disc21_validation_with_synthetic_data(self):
        splits = {"index": ["R000001", "R000002", "R000003"],
                  "val_targets": ["R000001"], "test_targets": ["R000002"],
                  "val_distractors": ["R000004"], "test_distractors": ["R000005"],
                  "dev_calibration": ["Q00000", "Q00001"], "dev_report": ["Q00002", "Q00003"]}
        truth = {"Q00000": "R000001", "Q00002": "R000002"}
        expected_counts = {n: len(ids) for n, ids in splits.items()}
        with patch("near_duplicates.validate.load_split", side_effect=lambda d, n: splits[n]), \
             patch("near_duplicates.validate.split_exists", return_value=False), \
             patch("near_duplicates.validate.dev_ground_truth", return_value=truth), \
             patch("near_duplicates.validate.fit_image_ids", return_value=["R000003"]), \
             patch("near_duplicates.validate.image_path", return_value=self.source):
            report = validate_dataset("disc21", expected_counts=expected_counts)
            self.assertEqual(report["unique_images"], 9)
            self.assertEqual(report["development_balance"]["dev_report"], {"matched": 1, "unmatched": 1})
            truth.update({"Q00001": "R000001", "Q00003": "R000002"})
            with self.assertRaisesRegex(ValueError, "equal matched and unmatched"):
                validate_dataset("disc21", expected_counts=expected_counts)
            del truth["Q00001"], truth["Q00003"]
            splits["dev_report"] = ["Q00000", "Q00003"]
            with self.assertRaisesRegex(ValueError, "overlapping IDs"):
                validate_dataset("disc21", expected_counts=expected_counts)

    def test_validator_reports_missing_images_and_overlap(self):
        splits = {"index": ["cat/image_0001.jpg"], "val_targets": ["cat/image_0001.jpg"],
                  "test_targets": ["cat/image_0002.jpg"], "val_distractors": ["cat/image_0003.jpg"],
                  "test_distractors": ["cat/image_0004.jpg"]}
        with patch("near_duplicates.validate.load_split", side_effect=lambda d, n: splits[n]):
            with self.assertRaisesRegex(ValueError, "expected 6209 IDs"):
                validate_dataset("caltech101")
            with self.assertRaisesRegex(ValueError, "targets outside index"):
                validate_dataset("caltech101", expected_counts={n: len(ids) for n, ids in splits.items()})
            splits["index"].append("cat/image_0002.jpg")
            with patch("near_duplicates.validate.image_path", return_value=self.root / "missing.jpg"):
                with self.assertRaisesRegex(ValueError, "4 image failures"):
                    validate_dataset("caltech101", expected_counts={n: len(ids) for n, ids in splits.items()})


if __name__ == "__main__":
    unittest.main()
