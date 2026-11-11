"""Deterministic preprocessing shared by catalogue images and queries."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


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
SIFT_PREPROCESS = PreprocessConfig(mode="RGB", max_side=512)


def load_image(path: str | Path, config: PreprocessConfig = SIFT_PREPROCESS) -> np.ndarray:
    """Decode an image or raise an error naming its path; never silently skip it."""
    try:
        with Image.open(path) as source:
            return preprocess_image(source, config)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot decode/preprocess {path}: {exc}") from exc


def preprocess_image(image: Image.Image, config: PreprocessConfig = SIFT_PREPROCESS) -> np.ndarray:
    """Apply the same pipeline to an in-memory transformed query as to a file."""
    image = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    image = Image.alpha_composite(background, image).convert(config.mode)
    if config.square_size is not None:
        image = image.resize((config.square_size, config.square_size), Image.Resampling.LANCZOS)
    elif config.max_side is not None:
        image.thumbnail((config.max_side, config.max_side), Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.uint8)
