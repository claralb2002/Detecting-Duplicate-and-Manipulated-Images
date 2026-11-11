from __future__ import annotations

import imagehash
import numpy as np
from PIL import Image

from conftest import textured_image
from near_duplicates.features import FeatureSet, image_features, load_features, save_features, source_image
from near_duplicates.local_features import extract_local_features
from near_duplicates.phash import phash
from near_duplicates.preprocessing import PHASH_PREPROCESS, preprocess_image


def bits(code: np.uint64) -> np.ndarray:
    return np.unpackbits(np.array([code], dtype=">u8").view(np.uint8)).astype(bool)


def hash_of(img: Image.Image) -> np.uint64:
    return phash(preprocess_image(img, PHASH_PREPROCESS))


def test_phash_matches_imagehash_on_synthetic_images():
    for seed in range(20):
        img = textured_image(seed)
        assert (bits(hash_of(img)) == imagehash.phash(img).hash.ravel()).all(), f"seed {seed}"


def test_phash_matches_imagehash_on_caltech(caltech_sample):
    images = [source_image(p) for p in caltech_sample]
    assert all((bits(hash_of(img)) == imagehash.phash(img).hash.ravel()).all() for img in images)


def test_phash_has_balanced_bits():
    """Thresholding at the median sets 32 of 64 bits unless coefficients tie at the median."""
    counts = [int(bits(hash_of(textured_image(seed))).sum()) for seed in range(10)]
    assert all(31 <= c <= 32 for c in counts)


def test_sift_descriptors_are_integers_in_byte_range(caltech_sample):
    """The uint8 cache is lossless only while OpenCV keeps saturating descriptor values to 0-255."""
    for path in caltech_sample[:10]:
        raw = extract_local_features(np.asarray(source_image(path)))["descriptors"]
        assert len(raw) and np.array_equal(raw, np.rint(raw)) and raw.min() >= 0 and raw.max() <= 255


def test_image_features_hash_the_full_image_and_find_keypoints_in_the_sift_copy():
    img = textured_image(0, size=(1200, 700))
    code, keypoints, descriptors = image_features(img)
    assert code == hash_of(img)
    assert descriptors.dtype == np.uint8 and len(keypoints) == len(descriptors) > 0
    assert keypoints[:, 0].max() < 512 and keypoints[:, 1].max() < round(700 * 512 / 1200)


def test_sift_keypoints_are_row_aligned_with_their_descriptors(caltech_sample):
    """Padding the image shifts every keypoint by the padding, so a descriptor found again sits 40 pixels further."""
    img = source_image(caltech_sample[0])
    _, keypoints, descriptors = image_features(img)
    assert (keypoints >= 0).all() and (keypoints[:, 0] < img.width).all() and (keypoints[:, 1] < img.height).all()
    padded = Image.new("RGB", (img.width + 80, img.height + 80), (127, 127, 127))
    padded.paste(img, (40, 40))
    _, moved, moved_descriptors = image_features(padded)
    distances = np.linalg.norm(descriptors[:, None].astype(float) - moved_descriptors[None].astype(float), axis=2)
    nearest = distances.argmin(axis=1)
    close = distances.min(axis=1) < 30  # descriptors found again, almost unchanged
    assert close.sum() >= 20
    assert np.median(np.abs(moved[nearest[close]] - keypoints[close] - 40)) < 1.0


def feature_set(ids: list[str], seed: int) -> FeatureSet:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(0, 5, size=len(ids))
    descriptors = [rng.integers(0, 256, size=(s, 128), dtype=np.uint8) for s in sizes]
    keypoints = [(rng.random((s, 2)) * 512).astype(np.float32) for s in sizes]
    return FeatureSet(ids, rng.integers(0, 2**64, size=len(ids), dtype=np.uint64), descriptors, keypoints)


def test_feature_cache_round_trip_keeps_image_alignment(tmp_path):
    ids = ["a/1.jpg", "b/2.jpg", "c/3.jpg", "d/4.jpg"]
    original = feature_set(ids, 0)
    original.descriptors[1] = np.zeros((0, 128), np.uint8)  # an image without keypoints must not shift the others
    original.keypoints[1] = np.zeros((0, 2), np.float32)
    save_features(tmp_path / "f.npz", original)
    loaded = load_features(tmp_path / "f.npz", ids)
    assert loaded is not None and loaded.ids == ids
    assert (loaded.hashes == original.hashes).all()
    assert all(np.array_equal(a, b) for a, b in zip(loaded.descriptors, original.descriptors))
    assert all(np.array_equal(a, b) for a, b in zip(loaded.keypoints, original.keypoints))


def test_feature_cache_rejects_other_image_lists(tmp_path):
    save_features(tmp_path / "f.npz", feature_set(["x", "y"], 0))
    assert load_features(tmp_path / "f.npz", ["y", "x"]) is None
    assert load_features(tmp_path / "missing.npz", ["x", "y"]) is None
