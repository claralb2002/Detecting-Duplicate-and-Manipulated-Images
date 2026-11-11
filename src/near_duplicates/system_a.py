"""Reproducible System A validation and parameter tuning, using only val_* splits.

Run: python -m near_duplicates.system_a caltech101
No test or dev_report query is loaded. This is calibration, not final evaluation.
"""

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
import json
from pathlib import Path
import platform

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, __version__ as pillow_version
import scipy
from scipy.fftpack import dct

from near_duplicates.datasets import DATASETS, SPLIT_SEED, image_path, load_split
from near_duplicates.lsh import BitSamplingLSH, collision_probability
from near_duplicates.phash import PHASH_PARAMETERS, hash_array, hash_split, hamming_distances, phash
from near_duplicates.preprocessing import PHASH_PREPROCESS, preprocess_image

TRANSFORMS = ("jpeg_50", "resize_half", "brightness_115", "crop_10", "rotate_5")


@dataclass(frozen=True)
class ValidationQueries:
    hashes: np.ndarray
    expected_ids: tuple[str | None, ...]
    transforms: tuple[str, ...]


def _transform(image: Image.Image, name: str) -> Image.Image:
    """Fixed calibration probes, applied before 32x32 pHash preprocessing."""
    if name == "jpeg_50":
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=50)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return encoded.copy()
    if name == "resize_half":
        return image.resize((max(1, image.width // 2), max(1, image.height // 2)), Image.Resampling.LANCZOS)
    if name == "brightness_115":
        return ImageEnhance.Brightness(image).enhance(1.15)
    if name == "crop_10":
        dx, dy = int(image.width * 0.05), int(image.height * 0.05)
        return image.crop((dx, dy, image.width - dx, image.height - dy))
    if name == "rotate_5":
        return image.rotate(5, resample=Image.Resampling.BICUBIC, fillcolor="white")
    raise ValueError(f"unknown transformation {name!r}")


def make_validation_queries(dataset: str, *, limit: int | None = None) -> ValidationQueries:
    """Use every validation source by default; a limit is a seeded smoke subset."""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    hashes, expected, transforms = [], [], []
    rng = np.random.default_rng(SPLIT_SEED)
    for split in ("val_targets", "val_distractors"):
        ids = load_split(dataset, split)
        if limit is not None:
            ids = [ids[i] for i in sorted(rng.choice(len(ids), min(limit, len(ids)), replace=False))]
        for image_id in ids:
            with Image.open(image_path(dataset, image_id)) as source:
                rgba = ImageOps.exif_transpose(source).convert("RGBA")
                image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
            for name in TRANSFORMS:
                hashes.append(phash(preprocess_image(_transform(image, name), PHASH_PREPROCESS)))
                expected.append(image_id if split == "val_targets" else None)
                transforms.append(name)
    return ValidationQueries(np.array(hashes, dtype=np.uint64), tuple(expected), tuple(transforms))


def _metrics(predicted: np.ndarray, distances: np.ndarray, expected: np.ndarray, radius: int) -> dict:
    accepted = (predicted >= 0) & (distances <= radius)
    positive = expected >= 0
    correct = accepted & positive & (predicted == expected)
    tp = int(correct.sum())
    fp = int(accepted.sum()) - tp  # Includes a wrong source returned for a positive query.
    fn = int(positive.sum()) - tp
    negative_count = int((~positive).sum())
    return {
        "queries": len(expected), "positive_queries": int(positive.sum()),
        "negative_queries": negative_count, "true_positives": tp,
        "false_positives": fp, "false_negatives": fn,
        "recall_at_1": tp / int(positive.sum()) if positive.any() else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "distractor_false_positive_rate": float((accepted & ~positive).sum()) / negative_count if negative_count else 0.0,
    }


def tune_lsh(
    hashes: Sequence[int] | np.ndarray, image_ids: Sequence[str], queries: ValidationQueries, *,
    b_values: Sequence[int] = (8, 16, 32), r_values: Sequence[int] = (8, 12, 16),
    thresholds: Sequence[int] = tuple(range(17)), seed: int = SPLIT_SEED,
) -> dict:
    """Sweep tables, bits/table and radius; select on validation F1.

    Tie-breaks: recall, fewer average candidates, fewer bucket entries (b),
    larger r, then smaller radius. No timing-dependent selection.
    Exact all-catalogue nearest neighbours diagnose LSH losses separately from
    pHash/threshold losses. This baseline only supports System A calibration.
    """
    hashes = hash_array(hashes)
    qhashes = hash_array(queries.hashes)
    if not len(hashes) or not len(qhashes):
        raise ValueError("tuning requires nonempty catalogue and queries")
    if len(qhashes) != len(queries.expected_ids) or len(qhashes) != len(queries.transforms):
        raise ValueError("each query needs a hash, expected ID and transformation")
    if not b_values or not r_values or not thresholds:
        raise ValueError("parameter grids must be nonempty")
    if any(type(t) is not int or not 0 <= t <= 64 for t in thresholds):
        raise ValueError("thresholds must be integers in [0, 64]")
    # Validate catalogue and construction parameters even before the first sweep.
    BitSamplingLSH(hashes, image_ids, b=b_values[0], r=r_values[0], seed=seed)
    lookup = {image_id: row for row, image_id in enumerate(image_ids)}
    if any(i is not None and i not in lookup for i in queries.expected_ids):
        raise ValueError("every positive query source must be in the catalogue")
    expected = np.array([lookup[i] if i is not None else -1 for i in queries.expected_ids])
    if not (expected >= 0).any() or not (expected < 0).any():
        raise ValueError("tuning requires both positive and distractor queries")
    # Alphabetical ID order makes ties agree with BitSamplingLSH.query().
    id_order = np.argsort(np.asarray(image_ids), kind="stable")
    ranks = np.empty(len(hashes), dtype=np.intp)
    ranks[id_order] = np.arange(len(hashes))
    all_distances = np.empty((len(qhashes), len(hashes)), dtype=np.uint8)
    exact_rows, exact_distances, source_distances = [], [], []
    for i, (query, source) in enumerate(zip(qhashes, expected)):
        distances = hamming_distances(query, hashes)
        nearest = int(id_order[np.argmin(distances[id_order])])
        exact_rows.append(nearest)
        exact_distances.append(int(distances[nearest]))
        source_distances.append(int(distances[source]) if source >= 0 else -1)
        all_distances[i] = distances
    # Q*N uint8 distances make a sweep cheap at calibration scale (5000*6209 ~31 MB).
    # No catalogue self-distance N*N matrix is constructed.
    exact_rows, exact_distances = np.array(exact_rows), np.array(exact_distances)
    best_key, best, best_diagnostics = None, None, None
    sweep = []
    for b in b_values:
        for r in r_values:
            index = BitSamplingLSH(hashes, image_ids, b=b, r=r, seed=seed)
            predicted = np.full(len(qhashes), -1, dtype=np.intp)
            distances = np.full(len(qhashes), 65, dtype=np.uint8)
            candidate_counts, retrieved_sources, retrieved_exact = [], 0, 0
            for i, query in enumerate(qhashes):
                rows = index.candidates(query)
                candidate_counts.append(len(rows))
                retrieved_sources += int(expected[i] >= 0 and expected[i] in rows)
                retrieved_exact += int(exact_rows[i] in rows)
                if len(rows):
                    order = np.lexsort((ranks[rows], all_distances[i, rows]))
                    predicted[i] = rows[order[0]]
                    distances[i] = all_distances[i, predicted[i]]
            diagnostics = {
                "mean_candidates": float(np.mean(candidate_counts)),
                "candidate_fraction": float(np.mean(candidate_counts)) / len(hashes),
                "source_candidate_recall": retrieved_sources / int((expected >= 0).sum()),
                "exact_nearest_candidate_recall": retrieved_exact / len(qhashes),
            }
            for radius in thresholds:
                metrics = _metrics(predicted, distances, expected, radius)
                row = {"b": b, "r": r, "max_distance": radius, **metrics, **diagnostics}
                sweep.append(row)
                key = (metrics["f1"], metrics["recall_at_1"], -diagnostics["mean_candidates"], -b, r, -radius)
                if best_key is None or key > best_key:
                    best_key, best = key, row
                    groups = {}
                    for name in sorted(set(queries.transforms)):
                        selected = np.array(queries.transforms) == name
                        groups[name] = _metrics(predicted[selected], distances[selected], expected[selected], radius)
                    best_diagnostics = {
                        "by_transform": groups,
                        "exhaustive_at_selected_threshold": _metrics(exact_rows, exact_distances, expected, radius),
                    }
    source_distances = np.array(source_distances)
    hash_validation = {}
    for name in sorted(set(queries.transforms)):
        values = source_distances[(np.array(queries.transforms) == name) & (expected >= 0)]
        hash_validation[name] = {
            "positive_source_hamming_median": float(np.median(values)),
            "positive_source_hamming_p95": float(np.percentile(values, 95)),
            "positive_source_hamming_max": int(values.max()),
        }
    return {
        "selected": best, **best_diagnostics, "hash_validation": hash_validation,
        "sweep": sweep,
        "theoretical_candidate_probabilities": [
            {"distance": d, "probability": collision_probability(d, best["b"], best["r"])}
            for d in range(65)
        ],
    }


def validate_dct_reference(dataset: str, image_ids: Sequence[str], hashes: np.ndarray) -> dict:
    """Cross-check the extraction against separable legacy SciPy DCT-II calls.

    Unit tests also use an explicit cosine-matrix DCT independent of SciPy's FFT.
    Comparison starts from our shared preprocessed pixels, so tests of EXIF/alpha
    remain the responsibility of preprocessing tests.
    """
    mismatches = []
    from near_duplicates.preprocessing import load_image

    for image_id, actual in zip(image_ids, hashes):
        pixels = load_image(image_path(dataset, image_id), PHASH_PREPROCESS)
        low = dct(dct(pixels.astype(np.float64), axis=0), axis=1)[:8, :8]
        expected = int("".join("1" if v > np.median(low) else "0" for v in low.ravel()), 2)
        if expected != int(actual):
            mismatches.append(image_id)
    if mismatches:
        raise ValueError(f"reference pHash disagrees on {len(mismatches)} images: {mismatches[:5]}")
    return {"images_checked": len(image_ids), "mismatches": 0, "reference": "separable scipy.fftpack.dct, float64"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--limit", type=int, help="seeded smoke subset per validation source split; still uses full index")
    parser.add_argument("--output", type=Path, default=Path("results/system_a_calibration.json"))
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    image_ids = load_split(args.dataset, "index")
    print(f"Hashing {len(image_ids)} catalogue images (cached).", flush=True)
    hashes = hash_split(args.dataset, image_ids)
    # Seeded reference sample from the catalogue; no test query sources are loaded separately.
    rows = np.sort(np.random.default_rng(SPLIT_SEED).choice(len(image_ids), min(100, len(image_ids)), replace=False))
    reference = validate_dct_reference(args.dataset, [image_ids[i] for i in rows], hashes[rows])
    print("Generating validation targets and distractors before pHash preprocessing.", flush=True)
    queries = make_validation_queries(args.dataset, limit=args.limit)
    print(f"Tuning 9 (b, r) settings and radii 0..16 on {len(queries.hashes)} validation queries.", flush=True)
    result = tune_lsh(hashes, image_ids, queries)
    report = {
        "dataset": args.dataset, "stage": "validation_calibration", "seed": SPLIT_SEED,
        "source_splits": ["index", "val_targets", "val_distractors"],
        "source_limit_per_split": args.limit, "catalogue_size": len(image_ids),
        "transforms": list(TRANSFORMS), "phash": PHASH_PARAMETERS,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pillow": pillow_version},
        "selection": "max validation F1, then recall, fewer candidates, fewer tables, larger r, smaller radius",
        "dct_reference": reference, **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["selected"], indent=2))
    print(f"Calibration saved to {args.output}; test and dev_report remain reserved.")


if __name__ == "__main__":
    main()
