"""Per-image numeric feature caches; feature extraction is supplied by the caller."""

from collections.abc import Callable
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np
from PIL import __version__ as pillow_version

from near_duplicates.datasets import DATA_DIR, image_path
from near_duplicates.preprocessing import PreprocessConfig, load_image


class FeatureCache:
    """Cache named arrays, including variable-length descriptors/keypoints.

    Include extractor version, settings, seeds and fitted-vocabulary identity in
    ``parameters``. Increment ``schema`` if preprocessing semantics change.
    Source content, preprocessing and library versions are included automatically.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else DATA_DIR / "cache"

    def get_or_compute(
        self, dataset: str, image_id: str, config: PreprocessConfig,
        feature: str, parameters: dict,
        compute: Callable[[np.ndarray], dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        return self.get_or_compute_derived(
            dataset, image_id, config, feature, parameters,
            lambda: compute(load_image(image_path(dataset, image_id), config)),
        )

    def get_or_compute_derived(
        self, dataset: str, image_id: str, config: PreprocessConfig,
        feature: str, parameters: dict,
        compute: Callable[[], dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Cache derived arrays without decoding an image on a cache hit.

        Source-content invalidation is unchanged. Callers may lazily share one
        decoded source across multiple extractors or transformed queries.
        Include the full derivation/transform identity in parameters.
        """
        source = image_path(dataset, image_id)
        metadata = json.dumps({
            "schema": 1, "dataset": dataset, "image_id": image_id,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "preprocessing": asdict(config), "feature": feature,
            "parameters": parameters, "pillow": pillow_version, "numpy": np.__version__,
        }, sort_keys=True, allow_nan=False, separators=(",", ":"))
        key = hashlib.sha256(metadata.encode()).hexdigest()
        target = self.root / key[:2] / f"{key}.npz"
        if target.exists():
            try:
                with np.load(target, allow_pickle=False) as cached:
                    if cached["__metadata__"].item() == metadata:
                        arrays = {name: cached[name] for name in cached.files if name != "__metadata__"}
                        if arrays:
                            return arrays
            except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
                pass  # A damaged/interrupted cache is regenerated, not treated as data.

        arrays = compute()
        if not arrays or any(
            not isinstance(name, str) or not name.isidentifier() or name.startswith("__")
            or name in {"file", "allow_pickle"}
            or not isinstance(value, np.ndarray) or value.dtype.kind not in "buifc"
            for name, value in arrays.items()
        ):
            raise ValueError("features must be a nonempty mapping of ordinary names to numeric numpy arrays")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, __metadata__=np.array(metadata), **arrays)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return arrays
