from __future__ import annotations

import numpy as np
import pytest

from near_duplicates.transforms import CONDITIONS, query_seed, transform


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def pixels(img) -> np.ndarray:
    return np.asarray(img, dtype=float)


@pytest.mark.parametrize("kind,strength", CONDITIONS)
def test_every_condition_returns_rgb_and_is_deterministic(image, kind, strength):
    first = transform(image, kind, strength, rng(query_seed(kind, strength, "x")))
    second = transform(image, kind, strength, rng(query_seed(kind, strength, "x")))
    assert first.mode == "RGB"
    assert np.array_equal(pixels(first), pixels(second))


def test_query_seed_is_stable_across_processes():
    """Python's hash() is salted per process; the seed must not be, or cached queries would not match reruns."""
    assert query_seed("crop", 0.5, "accordion/image_0001.jpg") == 2478112141


def test_crop_keeps_the_requested_area_as_a_subimage(image):
    w, h = image.size
    for strength in (0.9, 0.5, 0.3):
        out = transform(image, "crop", strength, rng())
        assert abs(out.size[0] * out.size[1] / (w * h) - strength) < 0.01
        source = pixels(image)
        crop = pixels(out)
        offsets = [(x, y) for y in range(h - crop.shape[0] + 1) for x in range(w - crop.shape[1] + 1)
                   if np.array_equal(source[y, x], crop[0, 0])]
        assert any(np.array_equal(source[y : y + crop.shape[0], x : x + crop.shape[1]], crop) for x, y in offsets)


def test_downscale_sets_each_side(image):
    out = transform(image, "downscale", 0.25, rng())
    assert out.size == (round(image.size[0] * 0.25), round(image.size[1] * 0.25))


def test_overlay_covers_the_requested_share_of_rows(image):
    for strength in (0.05, 0.15, 0.3):
        changed_rows = (pixels(transform(image, "overlay", strength, rng())) != pixels(image)).any(axis=(1, 2))
        assert changed_rows.sum() == round(image.size[1] * strength)


@pytest.mark.parametrize("kind", ["jpeg", "blur", "brightness", "contrast"])
def test_distortion_grows_with_strength(image, kind):
    strengths = [s for k, s in CONDITIONS if k == kind]
    if kind == "jpeg":
        strengths = sorted(strengths, reverse=True)  # lower JPEG quality is stronger
    errors = [np.abs(pixels(transform(image, kind, s, rng())) - pixels(image)).mean() for s in strengths]
    assert errors == sorted(errors) and errors[0] > 0


def test_flip_mirrors_horizontally(image):
    assert np.array_equal(pixels(transform(image, "flip", 1, rng())), pixels(image)[:, ::-1])


def test_unknown_kind_is_rejected(image):
    with pytest.raises(ValueError):
        transform(image, "sepia", 1, rng())
