"""Resumable joint pHash/SIFT extraction from one decoded RGB source."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Callable, Sequence

import cv2
import numpy as np
from PIL import Image

from near_duplicates.cache import FeatureCache
from near_duplicates.datasets import image_path
from near_duplicates.phash import PHASH_PARAMETERS, phash
from near_duplicates.preprocessing import PHASH_PREPROCESS, PreprocessConfig, preprocess_image

RGB_SOURCE = PreprocessConfig(mode="RGB")


@dataclass(frozen=True)
class SIFTConfig:
    max_features: int = 300
    max_side: int = 512
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.max_features, self.max_side)):
            raise ValueError("SIFT sizes and feature cap must be positive integers")
        if any(not np.isfinite(v) or v <= 0 for v in (self.contrast_threshold, self.edge_threshold, self.sigma)):
            raise ValueError("SIFT thresholds and sigma must be finite and positive")

    @property
    def parameters(self) -> dict:
        return {"version": 1, "sift": asdict(self), "opencv": cv2.__version__, "phash": PHASH_PARAMETERS}


def extract_local_features(rgb: np.ndarray, config: SIFTConfig = SIFTConfig()) -> dict[str, np.ndarray]:
    """Retain at most 300 strongest keypoints, including all cached geometry.

    Coordinates refer to the aspect-preserving resized SIFT image. pHash uses
    the original decoded pixels, not a second resize of the SIFT thumbnail.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("expected a nonempty RGB uint8 image")
    source = Image.fromarray(rgb)
    resized = preprocess_image(source, PreprocessConfig(mode="RGB", max_side=config.max_side))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=config.max_features, contrastThreshold=config.contrast_threshold,
                           edgeThreshold=config.edge_threshold, sigma=config.sigma)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    order = sorted(range(len(keypoints)), key=lambda i: (
        -keypoints[i].response, *keypoints[i].pt, keypoints[i].size, keypoints[i].angle, keypoints[i].octave,
    ))[:config.max_features]
    count = len(order)
    return {
        "hashes": np.array([phash(preprocess_image(source, PHASH_PREPROCESS))], dtype=np.uint64),
        "descriptors": np.ascontiguousarray(descriptors[order], dtype=np.float32) if count else np.empty((0, 128), np.float32),
        "keypoints": np.array([keypoints[i].pt for i in order], dtype=np.float32).reshape(-1, 2),
        "responses": np.array([keypoints[i].response for i in order], dtype=np.float32),
        "offsets": np.array([0, count], dtype=np.int64),
        "image_shape": np.array(gray.shape, dtype=np.int32),
    }


class LocalFeatureStore:
    """Numeric per-source caches; changing pixels or extractor settings invalidates them."""

    def __init__(self, dataset: str, config: SIFTConfig = SIFTConfig(), cache: FeatureCache | None = None):
        self.dataset, self.config = dataset, config
        self.cache = cache if cache is not None else FeatureCache()

    def get(self, image_id: str) -> dict[str, np.ndarray]:
        return self.cache.get_or_compute(self.dataset, image_id, RGB_SOURCE, "joint-phash-sift",
                                         self.config.parameters, lambda rgb: extract_local_features(rgb, self.config))

    def transformed(self, image_id: str, transform: dict, compute_rgb: Callable[[], np.ndarray]) -> dict[str, np.ndarray]:
        return self.cache.get_or_compute_derived(
            self.dataset, image_id, RGB_SOURCE, "joint-phash-sift-query",
            {**self.config.parameters, "transform": transform},
            lambda: extract_local_features(compute_rgb(), self.config),
        )

    def fingerprint(self, image_ids: Sequence[str]) -> str:
        digest = hashlib.sha256(json.dumps(self.config.parameters, sort_keys=True).encode())
        for image_id in image_ids:
            digest.update(image_id.encode() + b"\0")
            digest.update(hashlib.sha256(image_path(self.dataset, image_id).read_bytes()).digest())
        return digest.hexdigest()


def save_arrays(path: str | Path, metadata: dict, **arrays: np.ndarray) -> None:
    """Atomically persist a numeric artifact with JSON metadata; no pickle/joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, metadata=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)), **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_feature_bundle(path: str | Path, image_ids: Sequence[str], get_features: Callable[[str], dict]) -> None:
    """Export a bounded shard in the gameplan's hashes/descriptors/offsets layout.

    Use shards for large datasets: this convenience exporter holds one shard in
    memory. Retrieval keeps only words/signatures and loads descriptors lazily.
    """
    features = [get_features(i) for i in image_ids]
    counts = [len(f["descriptors"]) for f in features]
    save_arrays(path, {"image_ids": list(image_ids), "schema": 1},
                hashes=np.concatenate([f["hashes"] for f in features]) if features else np.empty(0, np.uint64),
                descriptors=np.concatenate([f["descriptors"] for f in features]) if features else np.empty((0, 128), np.float32),
                keypoints=np.concatenate([f["keypoints"] for f in features]) if features else np.empty((0, 2), np.float32),
                offsets=np.array([0, *np.cumsum(counts)], dtype=np.int64))
