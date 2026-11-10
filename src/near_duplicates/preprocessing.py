"""Deterministic preprocessing shared by catalogue images and queries."""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

import numpy as np
from PIL import Image, ImageOps

from near_duplicates.datasets import image_path, load_split


@dataclass(frozen=True)
class PreprocessConfig:
    """Return uint8 pixels without normalization; resize with Lanczos.

    ``square_size`` stretches to a square (pHash); ``max_side`` preserves
    aspect ratio and only downsizes (local features). EXIF orientation is applied
    before conversion. Transparent pixels are composited onto white.
    """

    mode: str = "L"
    square_size: int | None = None
    max_side: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("L", "RGB"):
            raise ValueError("mode must be L or RGB")
        for size in (self.square_size, self.max_side):
            if size is not None and (type(size) is not int or size <= 0):
                raise ValueError("image sizes must be positive integers")
        if self.square_size is not None and self.max_side is not None:
            raise ValueError("choose square_size or max_side, not both")


PHASH_PREPROCESS = PreprocessConfig(square_size=32)
SIFT_PREPROCESS = PreprocessConfig(max_side=1024)


def load_image(path: str | Path, config: PreprocessConfig = SIFT_PREPROCESS) -> np.ndarray:
    """Decode an image or raise an error naming its path; never silently skip it."""
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            image = Image.alpha_composite(background, image).convert(config.mode)
            if config.square_size is not None:
                image = image.resize((config.square_size, config.square_size), Image.Resampling.LANCZOS)
            elif config.max_side is not None:
                image.thumbnail((config.max_side, config.max_side), Image.Resampling.LANCZOS)
            return np.array(image, dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot decode/preprocess {path}: {exc}") from exc


def iter_images(
    dataset: str, split: str, config: PreprocessConfig = SIFT_PREPROCESS,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield image IDs and pixels in committed split order, one image at a time."""
    for image_id in load_split(dataset, split):
        yield image_id, load_image(image_path(dataset, image_id), config)


def iter_batches(
    dataset: str, split: str, config: PreprocessConfig = SIFT_PREPROCESS,
    batch_size: int = 32,
) -> Iterator[list[tuple[str, np.ndarray]]]:
    """Yield bounded batches; lists support images with different aspect ratios."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    batch = []
    for item in iter_images(dataset, split, config):
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
