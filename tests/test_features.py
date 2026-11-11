from __future__ import annotations

import cv2
import imagehash
import numpy as np
from PIL import Image

from conftest import textured_image
from near_duplicates.features import (
    MAX_SIDE, SIFT_FEATURES, FeatureSet, load_features, load_image, phash, save_features, sift_features,
)


def bits(code: np.uint64) -> np.ndarray:
    return np.unpackbits(np.array([code], dtype=">u8").view(np.uint8)).astype(bool)


def test_phash_matches_imagehash_on_synthetic_images():
    for seed in range(20):
        img = textured_image(seed)
        assert (bits(phash(img)) == imagehash.phash(img).hash.ravel()).all(), f"seed {seed}"


def test_phash_matches_imagehash_on_caltech(caltech_sample):
    mismatched = [p.name for p in caltech_sample if not (bits(phash(load_image(p))) == imagehash.phash(load_image(p)).hash.ravel()).all()]
    assert not mismatched


def test_phash_has_balanced_bits():
    """Thresholding at the median sets 32 of 64 bits unless coefficients tie at the median."""
    counts = [int(bits(phash(textured_image(seed))).sum()) for seed in range(10)]
    assert all(31 <= c <= 32 for c in counts)


def test_load_image_bounds_the_longer_side(tmp_path):
    path = tmp_path / "big.png"
    textured_image(0, size=(1200, 700)).convert("L").save(path)
    img = load_image(path)
    assert img.mode == "RGB" and max(img.size) == MAX_SIDE and img.size[1] == round(700 * MAX_SIDE / 1200)


def test_opencv_sift_descriptors_are_integers_in_byte_range(caltech_sample):
    """The uint8 cache is lossless only while OpenCV keeps saturating descriptor values to 0-255."""
    for path in caltech_sample[:10]:
        _, raw = cv2.SIFT_create(nfeatures=SIFT_FEATURES).detectAndCompute(np.asarray(load_image(path).convert("L")), None)
        assert raw is not None
        assert np.array_equal(raw, np.rint(raw)) and raw.min() >= 0 and raw.max() <= 255


def test_sift_keypoints_are_row_aligned_with_their_descriptors(caltech_sample):
    """Padding the image shifts every keypoint by the padding, so a descriptor found again sits 40 pixels further."""
    img = load_image(caltech_sample[0])
    keypoints, descriptors = sift_features(img)
    assert len(keypoints) == len(descriptors) > 0 and keypoints.dtype == np.float32
    assert (keypoints >= 0).all() and (keypoints[:, 0] < img.width).all() and (keypoints[:, 1] < img.height).all()
    padded = Image.new("RGB", (img.width + 80, img.height + 80), (127, 127, 127))
    padded.paste(img, (40, 40))
    moved, moved_descriptors = sift_features(padded)
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
