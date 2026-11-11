"""Cached validation probes for local-feature calibration; no test queries loaded."""

from dataclasses import dataclass
import hashlib
from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance

from near_duplicates.datasets import image_path, load_split
from near_duplicates.local_features import LocalFeatureStore, RGB_SOURCE
from near_duplicates.preprocessing import load_image


@dataclass(frozen=True)
class QuerySpec:
    source_id: str
    true_id: str | None
    kind: str
    strength: float
    seed: int

    @property
    def identity(self) -> str:
        return f"{self.source_id}|{self.kind}|{self.strength}|{self.seed}"

    @property
    def transform(self) -> dict:
        return {"version": 1, "kind": self.kind, "strength": self.strength, "seed": self.seed}


def transform_image(rgb: np.ndarray, spec: QuerySpec) -> np.ndarray:
    image = Image.fromarray(rgb)
    if spec.kind == "jpeg":
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=int(spec.strength))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            image = decoded.convert("RGB")
    elif spec.kind == "scale":
        image = image.resize((max(1, int(image.width * spec.strength)), max(1, int(image.height * spec.strength))),
                              Image.Resampling.LANCZOS)
    elif spec.kind == "crop":
        # Strength denotes retained AREA, with a seeded random crop position.
        side = np.sqrt(spec.strength)
        width, height = max(1, int(image.width * side)), max(1, int(image.height * side))
        rng = np.random.default_rng(spec.seed)
        x, y = int(rng.integers(image.width - width + 1)), int(rng.integers(image.height - height + 1))
        image = image.crop((x, y, x + width, y + height))
    elif spec.kind == "rotate":
        image = image.rotate(spec.strength, resample=Image.Resampling.BICUBIC, fillcolor="white")
    elif spec.kind == "brightness":
        image = ImageEnhance.Brightness(image).enhance(spec.strength)
    else:
        raise ValueError(f"unknown calibration probe {spec.kind!r}")
    return np.array(image)


def query_features(store: LocalFeatureStore, query: QuerySpec) -> dict:
    return store.transformed(query.source_id, query.transform,
                              lambda: transform_image(load_image(image_path(store.dataset, query.source_id), RGB_SOURCE), query))


def prepare_validation_queries(store: LocalFeatureStore, *, limit: int | None = None, seed: int = 2807) -> list[QuerySpec]:
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be positive")
    probes = (("jpeg", 50.0), ("scale", 0.5), ("crop", 0.75), ("rotate", 5.0), ("brightness", 1.15))
    queries = []
    rng = np.random.default_rng(seed)
    for split in ("val_targets", "val_distractors"):
        ids = load_split(store.dataset, split)
        if limit is not None and limit < len(ids):
            ids = [ids[i] for i in sorted(rng.choice(len(ids), limit, replace=False))]
        for image_id in ids:
            decoded = None

            def get_rgb(query):
                nonlocal decoded
                if decoded is None:
                    decoded = load_image(image_path(store.dataset, image_id), RGB_SOURCE)
                return transform_image(decoded, query)

            for kind, strength in probes:
                query_seed = int.from_bytes(hashlib.sha256(f"{seed}|{image_id}|{kind}|{strength}".encode()).digest()[:4], "little")
                query = QuerySpec(image_id, image_id if split == "val_targets" else None, kind, strength, query_seed)
                store.transformed(image_id, query.transform, lambda q=query: get_rgb(q))
                queries.append(query)
    return queries
