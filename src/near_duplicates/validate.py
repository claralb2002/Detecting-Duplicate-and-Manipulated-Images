"""Read-only dataset audit: python -m near_duplicates.validate caltech101."""

import argparse
import json
import re

from near_duplicates.datasets import (
    DATASETS, SPLIT_NAMES, CALTECH_EXCLUDED_CATEGORIES, dev_ground_truth,
    fit_image_ids, image_path, load_split, split_exists,
)
from near_duplicates.preprocessing import load_image, PreprocessConfig


EXPECTED_COUNTS = {
    "caltech101": {"index": 6209, "val_targets": 500, "test_targets": 2000,
                   "val_distractors": 500, "test_distractors": 2000},
    "disc21": {"index": 50_000, "val_targets": 500, "test_targets": 2000,
               "val_distractors": 500, "test_distractors": 2000,
               "dev_calibration": 10_000, "dev_report": 10_000, "scale": 50_000},
}


def validate_dataset(dataset: str, *, expected_counts: dict[str, int] | None = None) -> dict:
    """Check split structure, ground truth and every required image; raise on failure."""
    # The scale split is optional: it exists only where the 100,000-reference point was downloaded.
    splits = {name: load_split(dataset, name) for name in SPLIT_NAMES[dataset]
              if name != "scale" or split_exists(dataset, name)}
    expected = EXPECTED_COUNTS[dataset] if expected_counts is None else expected_counts
    errors = []
    for name, ids in splits.items():
        if name not in expected or len(ids) != expected[name]:
            errors.append(f"{name}: expected {expected.get(name)} IDs, found {len(ids)}")
        if not ids or len(ids) != len(set(ids)):
            errors.append(f"{name}: empty or duplicate IDs")
        for image_id in ids:
            if dataset == "disc21":
                prefix, digits, maximum = ("Q", 5, 50_000) if name.startswith("dev_") else ("R", 6, 1_000_000)
                valid = re.fullmatch(prefix + rf"\d{{{digits}}}", image_id) and int(image_id[1:]) < maximum
            else:
                valid = (re.fullmatch(r"[^/.]+/image_\d+\.jpg", image_id)
                         and image_id.split("/")[0] not in CALTECH_EXCLUDED_CATEGORIES)
            if not valid:
                errors.append(f"{name}: invalid image ID {image_id}")
    if errors:
        raise ValueError("\n".join(errors))
    index = set(splits["index"])
    for name in ("val_targets", "test_targets"):
        if not set(splits[name]) <= index:
            errors.append(f"{name}: targets outside index")
    for name in ("val_distractors", "test_distractors"):
        if set(splits[name]) & index:
            errors.append(f"{name}: distractors overlap index")
    for a, b in (("val_targets", "test_targets"), ("val_distractors", "test_distractors"),
                 ("dev_calibration", "dev_report")):
        if a in splits and set(splits[a]) & set(splits[b]):
            errors.append(f"{a}/{b}: overlapping IDs")
    report = {"dataset": dataset, "split_counts": {n: len(ids) for n, ids in splits.items()}}
    if dataset == "disc21":
        truth = dev_ground_truth()
        if not truth or any(not re.fullmatch(r"Q\d{5}", q) or int(q[1:]) >= 50_000
                            or not re.fullmatch(r"R\d{6}", r) or int(r[1:]) >= 1_000_000
                            for q, r in truth.items()):
            errors.append("invalid or empty development ground truth")
        if not set(truth.values()) <= index:
            errors.append("development ground-truth references missing from index")
        positives = set(truth)
        dev_ids = set(splits["dev_calibration"]) | set(splits["dev_report"])
        if not positives <= dev_ids:
            errors.append("positive development queries missing from calibration/report")
        report["development_balance"] = {
            n: {"matched": len(set(splits[n]) & positives), "unmatched": len(set(splits[n]) - positives)}
            for n in ("dev_calibration", "dev_report")
        }
        if report["development_balance"]["dev_calibration"] != report["development_balance"]["dev_report"]:
            errors.append("calibration/report matched-unmatched counts differ")
        for name, balance in report["development_balance"].items():
            if balance["matched"] != balance["unmatched"]:
                errors.append(f"{name}: expected equal matched and unmatched query counts")
    if errors:
        raise ValueError("\n".join(errors))
    ids = sorted(set().union(*(set(values) for values in splits.values())))
    for image_id in ids:
        path = image_path(dataset, image_id)
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        try:
            load_image(path, PreprocessConfig())
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError(f"{len(errors)} image failures; first 20:\n" + "\n".join(errors[:20]))
    report.update(unique_images=len(ids), vocabulary_fit_images=len(fit_image_ids(dataset)))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    args = parser.parse_args()
    try:
        print(json.dumps(validate_dataset(args.dataset), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Validation failed: {exc}\n")


if __name__ == "__main__":
    main()
