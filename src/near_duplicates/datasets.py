"""Dataset locations, the fixed experiment splits, and DISC21 ground truth.

Splits are lists of image IDs in `splits/<dataset>/<name>.txt`, committed to git so every member
uses identical data. Tune only on `val_*`; report only on `test_*` and `dev_report`.

Image IDs:
- caltech101: path relative to the category root, e.g. `accordion/image_0001.jpg`.
- disc21: `R000123` for a reference image, `Q00004` for a development query.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("NEAR_DUPLICATES_DATA", REPO_ROOT / "data"))
SPLITS_DIR = REPO_ROOT / "splits"

CALTECH_ROOT = DATA_DIR / "caltech101" / "101_ObjectCategories"
# Faces_easy holds tight crops of the Faces images, i.e. genuine near-duplicates that would corrupt ground truth.
CALTECH_EXCLUDED_CATEGORIES = {"Faces_easy"}

DISC21_DIR = DATA_DIR / "disc21"
DISC21_REFERENCE_COUNT = 1_000_000
DISC21_QUERY_COUNT = 50_000

SPLIT_SEED = 2807
VALIDATION_TARGETS = 500
TEST_TARGETS = 2_000
VALIDATION_DISTRACTORS = 500
TEST_DISTRACTORS = 2_000

DATASETS = ("caltech101", "disc21")


def image_path(dataset: str, image_id: str) -> Path:
    if dataset == "caltech101":
        return CALTECH_ROOT / image_id
    if dataset == "disc21":
        folder = "references" if image_id.startswith("R") else "queries"
        return DISC21_DIR / folder / f"{image_id}.jpg"
    raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")


def load_split(dataset: str, name: str) -> list[str]:
    if dataset not in DATASETS or name not in SPLIT_NAMES[dataset]:
        raise ValueError(f"unknown dataset/split: {dataset}/{name}")
    path = SPLITS_DIR / dataset / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `uv run python -m near_duplicates.download {dataset}`")
    return path.read_text().split()


SPLIT_NAMES = {
    "caltech101": ("index", "val_targets", "test_targets", "val_distractors", "test_distractors"),
    "disc21": ("index", "val_targets", "test_targets", "val_distractors", "test_distractors",
               "dev_calibration", "dev_report", "scale"),
}


def fit_image_ids(dataset: str) -> list[str]:
    """Vocabulary-fitting pool: catalogue images excluding evaluation sources.

    DISC21 also excludes sources of all positive development queries. This uses
    ground truth only to reserve sources, never to fit or tune representations.
    Returned IDs are sorted; no committed split is rewritten.
    """
    excluded = set(load_split(dataset, "val_targets")) | set(load_split(dataset, "test_targets"))
    if dataset == "disc21":
        excluded.update(dev_ground_truth().values())
    return sorted(set(load_split(dataset, "index")) - excluded)


def dev_ground_truth() -> dict[str, str]:
    """DISC21 development query ID -> the reference ID it was made from (positive queries only)."""
    truth = {}
    seen = set()
    with open(DISC21_DIR / "dev_ground_truth.csv", newline="", encoding="utf-8-sig") as handle:
        for line, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(f"development ground truth line {line}: expected two columns")
            query, reference = (value.strip() for value in row)
            if line == 1 and (query, reference) == ("query_id", "reference_id"):
                continue
            if not re.fullmatch(r"Q[0-9]{5}", query) or int(query[1:]) >= DISC21_QUERY_COUNT:
                raise ValueError(f"development ground truth line {line}: invalid query ID {query!r}")
            if reference and (not re.fullmatch(r"R[0-9]{6}", reference)
                              or int(reference[1:]) >= DISC21_REFERENCE_COUNT):
                raise ValueError(f"development ground truth line {line}: invalid reference ID {reference!r}")
            if query in seen:
                raise ValueError(f"duplicate development ground-truth query: {query}")
            seen.add(query)
            if reference:
                truth[query] = reference
    return truth


def caltech_image_ids() -> list[str]:
    return sorted(
        path.relative_to(CALTECH_ROOT).as_posix()
        for path in CALTECH_ROOT.rglob("*.jpg")
        if path.parent.name not in CALTECH_EXCLUDED_CATEGORIES
    )


def write_splits(dataset: str, splits: dict[str, list[str]]) -> None:
    folder = SPLITS_DIR / dataset
    folder.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (folder / f"{name}.txt").write_text("\n".join(ids) + "\n")


def split_exists(dataset: str, name: str) -> bool:
    return (SPLITS_DIR / dataset / f"{name}.txt").exists()


def splits_exist(dataset: str) -> bool:
    return split_exists(dataset, "index")


def make_caltech_splits(image_ids: list[str]) -> dict[str, list[str]]:
    """Held-out images become distractor sources; validation and test targets are disjoint subsets of the index."""
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = [image_ids[i] for i in rng.permutation(len(image_ids))]
    held_out_count = VALIDATION_DISTRACTORS + TEST_DISTRACTORS
    held_out, index = shuffled[:held_out_count], shuffled[held_out_count:]
    targets = [index[i] for i in rng.permutation(len(index))[: VALIDATION_TARGETS + TEST_TARGETS]]
    return checked({
        "index": sorted(index),
        "val_targets": sorted(targets[:VALIDATION_TARGETS]),
        "test_targets": sorted(targets[VALIDATION_TARGETS:]),
        "val_distractors": sorted(held_out[:VALIDATION_DISTRACTORS]),
        "test_distractors": sorted(held_out[VALIDATION_DISTRACTORS:]),
    })


def make_disc21_splits(ground_truth: dict[str, str], index_size: int, negative_queries: int) -> dict[str, list[str]]:
    """Index every reference a positive development query was made from, plus random others up to `index_size`.

    Distractor sources are references outside the index. Development queries are split in half:
    `dev_calibration` for setting thresholds, `dev_report` for the reported score.
    """
    rng = np.random.default_rng(SPLIT_SEED)
    matched = sorted(set(ground_truth.values()))
    if index_size < len(matched):
        raise ValueError(f"index_size must be at least {len(matched)} to hold every ground-truth reference")
    all_references = np.array([f"R{i:06d}" for i in range(DISC21_REFERENCE_COUNT)])
    others = np.setdiff1d(all_references, matched)
    chosen = rng.permutation(others)
    extra = index_size - len(matched)
    distractor_count = VALIDATION_DISTRACTORS + TEST_DISTRACTORS
    index = matched + chosen[:extra].tolist()
    distractors = chosen[extra : extra + distractor_count].tolist()
    targets = [index[i] for i in rng.permutation(len(index))[: VALIDATION_TARGETS + TEST_TARGETS]]

    positives = sorted(ground_truth)
    negatives_pool = np.setdiff1d([f"Q{i:05d}" for i in range(DISC21_QUERY_COUNT)], positives)
    negatives = rng.permutation(negatives_pool)[:negative_queries].tolist()
    positives = rng.permutation(positives).tolist()
    return checked({
        "index": sorted(index),
        "val_targets": sorted(targets[:VALIDATION_TARGETS]),
        "test_targets": sorted(targets[VALIDATION_TARGETS:]),
        "val_distractors": sorted(distractors[:VALIDATION_DISTRACTORS]),
        "test_distractors": sorted(distractors[VALIDATION_DISTRACTORS:]),
        "dev_calibration": sorted(positives[::2] + negatives[::2]),
        "dev_report": sorted(positives[1::2] + negatives[1::2]),
    })


def make_disc21_scale_split(taken: set[str], count: int) -> list[str]:
    """`count` further random references outside every existing split, to grow the index for the scale point."""
    free = np.setdiff1d([f"R{i:06d}" for i in range(DISC21_REFERENCE_COUNT)], sorted(taken))
    return sorted(np.random.default_rng(SPLIT_SEED + 1).permutation(free)[:count].tolist())


def checked(splits: dict[str, list[str]]) -> dict[str, list[str]]:
    index = set(splits["index"])
    for name in ("val_targets", "test_targets"):
        assert set(splits[name]) <= index, f"{name} must be drawn from the index"
    for name in ("val_distractors", "test_distractors"):
        assert not set(splits[name]) & index, f"{name} must lie outside the index"
    for a, b in (("val_targets", "test_targets"), ("val_distractors", "test_distractors"), ("dev_calibration", "dev_report")):
        if a in splits:
            assert not set(splits[a]) & set(splits[b]), f"{a} and {b} must be disjoint"
    return splits
