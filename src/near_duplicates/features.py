"""Image preprocessing, the 64-bit perceptual hash, SIFT descriptors, and cached batch extraction.

A feature cache is one `.npz` per image list with `ids`, `hashes`, `descriptors` (all images stacked,
uint8) and `offsets` (image i owns rows offsets[i]:offsets[i+1]), plus the settings that produced it.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from multiprocessing.pool import Pool
from pathlib import Path

import cv2
import numpy as np
import scipy.fft
from PIL import Image
from tqdm import tqdm

from near_duplicates.datasets import image_path
from near_duplicates.transforms import query_seed, transform

MAX_SIDE = 512
SIFT_FEATURES = 300
DESCRIPTOR_SIZE = 128
SETTINGS = json.dumps({"max_side": MAX_SIDE, "sift_features": SIFT_FEATURES, "phash": "imagehash-dct"})

Job = tuple[str, str, str, float]
"""(dataset, image ID, transformation kind, strength)."""


@dataclass
class FeatureSet:
    ids: list[str]
    hashes: np.ndarray
    descriptors: list[np.ndarray]

    def subset(self, positions: np.ndarray) -> FeatureSet:
        return FeatureSet([self.ids[i] for i in positions], self.hashes[positions], [self.descriptors[i] for i in positions])


def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return img


def phash(img: Image.Image) -> np.uint64:
    """pHash: 32x32 grayscale, unnormalised 2-D DCT-II, 8x8 low-frequency block thresholded at its median.

    Bit i (most significant first) is coefficient (i // 8, i % 8), matching `imagehash.phash(img).hash.ravel()`.
    """
    pixels = np.asarray(img.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low_frequencies = scipy.fft.dctn(pixels, type=2)[:8, :8]
    bits = (low_frequencies > np.median(low_frequencies)).ravel()
    return np.packbits(bits).view(">u8")[0].astype(np.uint64)


def sift_descriptors(img: Image.Image) -> np.ndarray:
    """At most about SIFT_FEATURES descriptors; OpenCV saturates each value to 0-255, so uint8 is lossless."""
    _, descriptors = cv2.SIFT_create(nfeatures=SIFT_FEATURES).detectAndCompute(np.asarray(img.convert("L")), None)
    if descriptors is None:
        return np.zeros((0, DESCRIPTOR_SIZE), np.uint8)
    return descriptors.astype(np.uint8)


def extract_job(job: Job) -> tuple[np.uint64, np.ndarray]:
    dataset, image_id, kind, strength = job
    rng = np.random.default_rng(query_seed(kind, strength, image_id))
    img = transform(load_image(image_path(dataset, image_id)), kind, strength, rng)
    return phash(img), sift_descriptors(img)


def init_worker() -> None:
    cv2.setNumThreads(1)


def save_features(path: Path, features: FeatureSet) -> None:
    """Write to a temporary file and rename, so an interrupted run never leaves a corrupt cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.concatenate([[0], np.cumsum([len(d) for d in features.descriptors])])
    descriptors = np.concatenate(features.descriptors) if features.descriptors else np.zeros((0, DESCRIPTOR_SIZE), np.uint8)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, ids=np.array(features.ids), hashes=features.hashes, descriptors=descriptors, offsets=offsets,
             settings=np.array(SETTINGS))
    os.replace(tmp, path)


def load_features(path: Path, ids: list[str]) -> FeatureSet | None:
    """The cached features, or None if the cache is missing or was made for other images or settings."""
    if not path.exists():
        return None
    with np.load(path) as cached:
        if str(cached["settings"]) != SETTINGS or cached["ids"].tolist() != ids:
            return None
        return FeatureSet(ids, cached["hashes"], np.split(cached["descriptors"], cached["offsets"][1:-1]))


def cached_features(path: Path, jobs: list[Job], pool: Pool, label: str) -> FeatureSet:
    ids = [image_id for _, image_id, _, _ in jobs]
    if (features := load_features(path, ids)) is not None:
        return features
    results = list(tqdm(pool.imap(extract_job, jobs, chunksize=8), total=len(jobs), desc=label, file=sys.stderr))
    features = FeatureSet(ids, np.array([h for h, _ in results], dtype=np.uint64), [d for _, d in results])
    save_features(path, features)
    return features
