from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from near_duplicates.datasets import image_path, load_split


def textured_image(seed: int, size: tuple[int, int] = (320, 240)) -> Image.Image:
    """Smooth gradients plus random shapes: structure for SIFT and a non-trivial DCT spectrum for pHash."""
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([127 + 100 * np.sin(xx / rng.uniform(10, 40) + c) * np.cos(yy / rng.uniform(10, 40)) for c in range(3)], axis=-1)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for _ in range(25):
        x, y = rng.integers(0, w - 20), rng.integers(0, h - 20)
        draw.ellipse((x, y, x + rng.integers(8, 60), y + rng.integers(8, 60)), fill=tuple(int(c) for c in rng.integers(0, 256, 3)))
    return img


@pytest.fixture
def image() -> Image.Image:
    return textured_image(0)


def caltech_paths(count: int) -> list:
    try:
        ids = load_split("caltech101", "index")
    except FileNotFoundError:
        return []
    paths = [image_path("caltech101", i) for i in ids[:: max(1, len(ids) // count)][:count]]
    return [p for p in paths if p.exists()]


@pytest.fixture
def caltech_sample() -> list:
    paths = caltech_paths(40)
    if not paths:
        pytest.skip("Caltech-101 not downloaded; run `uv run python -m near_duplicates.download caltech101`")
    return paths
