"""Controlled transformations that turn an image into a query whose correct answer is known by construction."""

from __future__ import annotations

import io
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

# Strength units per kind: JPEG quality; fraction of each side kept; fraction of the area kept;
# degrees; Gaussian sigma in pixels; brightness gain 1 + s; contrast gain 1 - s; fraction of the area covered.
# The grid extends past the plan's levels until each system breaks, so every curve shows its failure point.
SINGLE_CONDITIONS: tuple[tuple[str, float], ...] = (
    ("clean", 0),
    *(("jpeg", quality) for quality in (90, 70, 50, 30, 10, 5)),
    *(("downscale", side) for side in (0.75, 0.5, 0.25, 0.15, 0.1)),
    *(("crop", area) for area in (0.9, 0.75, 0.5, 0.3, 0.15)),
    *(("rotate", degrees) for degrees in (2, 5, 10, 15, 30, 45)),
    *(("blur", sigma) for sigma in (1, 2, 4, 8)),
    *(("brightness", gain) for gain in (0.25, 0.5, 1.0, 2.0)),
    *(("contrast", loss) for loss in (0.25, 0.5, 0.75, 0.9)),
    *(("overlay", area) for area in (0.05, 0.15, 0.3, 0.5)),
    ("flip", 1),
)

# Realistic re-uploads chain edits. Each chain pairs an edit that defeats the global hash (crop, rotation,
# flip) with one that defeats local features (compression, downscaling, blur); strength is the level 1-3.
CHAINS: dict[str, tuple[tuple[tuple[str, float], ...], ...]] = {
    "crop+jpeg+overlay": (
        (("crop", 0.9), ("jpeg", 70), ("overlay", 0.05)),
        (("crop", 0.75), ("jpeg", 30), ("overlay", 0.15)),
        (("crop", 0.5), ("jpeg", 10), ("overlay", 0.3)),
    ),
    "crop+downscale": ((("crop", 0.9), ("downscale", 0.75)), (("crop", 0.75), ("downscale", 0.5)), (("crop", 0.5), ("downscale", 0.25))),
    "rotate+blur": ((("rotate", 2), ("blur", 1)), (("rotate", 5), ("blur", 2)), (("rotate", 10), ("blur", 4))),
    "flip+jpeg": ((("flip", 1), ("jpeg", 70)), (("flip", 1), ("jpeg", 30)), (("flip", 1), ("jpeg", 10))),
}

CONDITIONS = SINGLE_CONDITIONS + tuple((chain, level) for chain, levels in CHAINS.items() for level in range(1, len(levels) + 1))
KINDS = tuple(dict.fromkeys(kind for kind, _ in CONDITIONS))
MIN_SIDE = 8
OVERLAY_TEXT = "SHARED 1,000,000 TIMES"


def query_seed(kind: str, strength: float, image_id: str) -> int:
    """A seed that depends only on the query, so caches and reruns agree regardless of order or process."""
    return zlib.crc32(f"{kind}|{strength}|{image_id}".encode())


def transform(img: Image.Image, kind: str, strength: float, rng: np.random.Generator) -> Image.Image:
    if kind in CHAINS:
        for step, step_strength in CHAINS[kind][int(strength) - 1]:
            img = transform(img, step, step_strength, rng)
        return img
    w, h = img.size
    if kind == "clean":
        return img
    if kind == "jpeg":
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=int(strength))
        return Image.open(buffer).convert("RGB")
    if kind == "downscale":
        return img.resize((max(MIN_SIDE, round(w * strength)), max(MIN_SIDE, round(h * strength))), Image.Resampling.BILINEAR)
    if kind == "crop":
        side = np.sqrt(strength)
        cw, ch = max(MIN_SIDE, round(w * side)), max(MIN_SIDE, round(h * side))
        x, y = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
        return img.crop((x, y, x + cw, y + ch))
    if kind == "rotate":
        return img.rotate(strength, resample=Image.Resampling.BICUBIC, expand=True)
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(strength))
    if kind == "brightness":
        return ImageEnhance.Brightness(img).enhance(1 + strength)
    if kind == "contrast":
        return ImageEnhance.Contrast(img).enhance(1 - strength)
    if kind == "overlay":
        # The caption is drawn inside a band-sized image, so it is clipped and the covered area is exactly `strength`.
        band_height = round(h * strength)
        band = Image.new("RGB", (w, band_height), tuple(int(c) for c in rng.integers(0, 256, 3)))
        ImageDraw.Draw(band).text((5, band_height // 3), OVERLAY_TEXT, fill=(255, 255, 255))
        out = img.copy()
        out.paste(band, (0, int(rng.integers(0, h - band_height + 1))))
        return out
    if kind == "flip":
        return ImageOps.mirror(img)
    raise ValueError(f"unknown transformation {kind!r}; expected one of {KINDS}")
