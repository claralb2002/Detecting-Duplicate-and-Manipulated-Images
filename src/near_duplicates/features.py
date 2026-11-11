"""Cached batch extraction of every image's pHash, SIFT keypoints and descriptors.

An image is decoded at full resolution with the shared preprocessing (`local_features.RGB_SOURCE`), edited
there by `transforms.transform`, as a re-upload edits the original, and handed to
`local_features.extract_local_features`, which hashes the edited pixels and runs SIFT on a copy of at most
512 pixels.

A feature cache is one `.npz` per image list with `ids`, `hashes`, `descriptors` (all images stacked,
uint8), `keypoints` (the (x, y) position of each descriptor, float32) and `offsets` (image i owns rows
offsets[i]:offsets[i+1] of both), plus the settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from multiprocessing.pool import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from near_duplicates.datasets import DATA_DIR, image_path
from near_duplicates.local_features import RGB_SOURCE, SIFTConfig, extract_local_features
from near_duplicates.preprocessing import load_image
from near_duplicates.transforms import query_seed, transform

DESCRIPTOR_SIZE = 128
SETTINGS = json.dumps({"extractor": SIFTConfig().parameters, "edits": "full resolution", "descriptors": "uint8"}, sort_keys=True)


def cache_dir(dataset: str) -> Path:
    """Feature caches and vocabularies for `dataset` under the current extractor settings.

    Caches made with other settings live in other folders, so two versions of the pipeline never overwrite
    each other's caches.
    """
    return DATA_DIR / "cache" / f"{dataset}-{hashlib.sha256(SETTINGS.encode()).hexdigest()[:8]}"


Job = tuple[str, str, str, float]
"""(dataset, image ID, transformation kind, strength)."""


@dataclass
class FeatureSet:
    ids: list[str]
    hashes: np.ndarray
    descriptors: list[np.ndarray]
    keypoints: list[np.ndarray]

    def subset(self, positions: np.ndarray) -> FeatureSet:
        return FeatureSet([self.ids[i] for i in positions], self.hashes[positions],
                          [self.descriptors[i] for i in positions], [self.keypoints[i] for i in positions])

    def concatenate(self, other: FeatureSet) -> FeatureSet:
        return FeatureSet(self.ids + other.ids, np.concatenate([self.hashes, other.hashes]),
                          self.descriptors + other.descriptors, self.keypoints + other.keypoints)


def source_image(path: Path) -> Image.Image:
    """The full-resolution RGB image, decoded by the shared preprocessing."""
    return Image.fromarray(load_image(path, RGB_SOURCE))


def image_features(img: Image.Image) -> tuple[np.uint64, np.ndarray, np.ndarray]:
    """pHash, keypoint positions and descriptors of one image.

    OpenCV saturates SIFT descriptor values to the integers 0-255, so storing them as uint8 is lossless.
    """
    features = extract_local_features(np.asarray(img.convert("RGB")))
    return features["hashes"][0], features["keypoints"], features["descriptors"].astype(np.uint8)


def extract_job(job: Job) -> tuple[np.uint64, np.ndarray, np.ndarray]:
    dataset, image_id, kind, strength = job
    rng = np.random.default_rng(query_seed(kind, strength, image_id))
    return image_features(transform(source_image(image_path(dataset, image_id)), kind, strength, rng))


def init_worker() -> None:
    cv2.setNumThreads(1)


def save_features(path: Path, features: FeatureSet) -> None:
    """Write to a temporary file and rename, so an interrupted run never leaves a corrupt cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.concatenate([[0], np.cumsum([len(d) for d in features.descriptors])])
    descriptors = np.concatenate(features.descriptors) if features.descriptors else np.zeros((0, DESCRIPTOR_SIZE), np.uint8)
    keypoints = np.concatenate(features.keypoints) if features.keypoints else np.zeros((0, 2), np.float32)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, ids=np.array(features.ids), hashes=features.hashes, descriptors=descriptors, keypoints=keypoints,
             offsets=offsets, settings=np.array(SETTINGS))
    os.replace(tmp, path)


def load_features(path: Path, ids: list[str]) -> FeatureSet | None:
    """The cached features, or None if the cache is missing or was made for other images or settings."""
    if not path.exists():
        return None
    with np.load(path) as cached:
        if str(cached["settings"]) != SETTINGS or cached["ids"].tolist() != ids:
            return None
        boundaries = cached["offsets"][1:-1]
        return FeatureSet(ids, cached["hashes"], np.split(cached["descriptors"], boundaries), np.split(cached["keypoints"], boundaries))


def cached_features(path: Path, jobs: list[Job], pool: Pool, label: str) -> FeatureSet:
    ids = [image_id for _, image_id, _, _ in jobs]
    if (features := load_features(path, ids)) is not None:
        return features
    results = list(tqdm(pool.imap(extract_job, jobs, chunksize=8), total=len(jobs), desc=label, file=sys.stderr))
    features = FeatureSet(ids, np.array([h for h, _, _ in results], dtype=np.uint64), [d for _, _, d in results],
                          [k for _, k, _ in results])
    save_features(path, features)
    return features
