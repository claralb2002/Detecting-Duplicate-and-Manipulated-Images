"""System B calibration: SIFT -> visual words -> MinHash LSH -> exact/RANSAC verification.

Only val_* queries are generated. Vocabulary fitting excludes validation/test
sources. Full runs use two seeds; --limit caps catalogue, fitting and query pools.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from functools import lru_cache
import csv
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
from zipfile import BadZipFile

import cv2
import numpy as np
import sklearn
from PIL import __version__ as pillow_version

from near_duplicates.datasets import DATASETS, SPLIT_SEED, fit_image_ids, load_split
from near_duplicates.local_features import LocalFeatureStore, save_arrays
from near_duplicates.local_queries import QuerySpec, prepare_validation_queries, query_features
from near_duplicates.minhash import MinHasher, MinHashLSH, candidate_probability
from near_duplicates.retrieval_metrics import retrieval_metrics
from near_duplicates.spatial import SpatialConfig, verify_spatial
from near_duplicates.visual_index import VisualWordIndex
from near_duplicates.visual_words import VisualVocabulary, fit_vocabulary, sample_descriptors

BANDS = ((64, 2), (32, 4), (16, 8), (8, 16))
THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
CSV_FIELDS = ("system", "mode", "kind", "strength", "true_id", "predicted", "score", "candidates",
              "true_is_candidate", "true_similarity", "lsh_seconds", "verification", "source_id", "seed",
              "vocabulary_size", "metric", "inliers")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _winner(scores: np.ndarray, rows: np.ndarray, ids: list[str]) -> int:
    if not len(rows):
        return -1
    best = scores[rows].max()
    if best <= 0:
        return -1
    tied = rows[scores[rows] == best]
    return int(min(tied, key=lambda row: ids[row]))


def sweep_words(sets, query_sets, ids, queries, hasher, *, bands=BANDS, thresholds=THRESHOLDS):
    """Tune exact set scoring, recording observed versus theoretical S-curves.

    Calibration reuses exhaustive set scores across band settings. Live queries
    use VisualWordIndex.query, which scores only their candidate rows.
    """
    index = VisualWordIndex(sets, ids, hasher, b=bands[0][0], r=bands[0][1])
    lshes = [MinHashLSH(index.signatures, ids, b=b, r=r) for b, r in bands]
    qsignatures = hasher.transform(query_sets)
    lookup = {image_id: row for row, image_id in enumerate(ids)}
    expected = np.array([lookup[q.true_id] if q.true_id is not None else -1 for q in queries], np.intp)
    count = len(queries)
    predictions = np.full((len(bands), 2, count), -1, np.intp)
    winning_scores = np.zeros((len(bands), 2, count), np.float32)
    exact_predictions = np.full((2, count), -1, np.intp)
    exact_scores = np.zeros((2, count), np.float32)
    candidate_counts = np.zeros((len(bands), count), np.intp)
    source_candidates = np.zeros((len(bands), count), bool)
    source_jaccard = np.zeros(count, np.float32)
    all_rows = np.flatnonzero(index.sizes > 0)
    for i, words in enumerate(query_sets):
        scores = [index.scores(words, metric=metric) for metric in ("jaccard", "containment")]
        if expected[i] >= 0:
            source_jaccard[i] = scores[0][expected[i]]
        for metric, values in enumerate(scores):
            row = _winner(values, all_rows, ids)
            exact_predictions[metric, i] = row
            exact_scores[metric, i] = values[row] if row >= 0 else 0
        for band, lsh in enumerate(lshes):
            rows = lsh.candidates(qsignatures[i])
            candidate_counts[band, i] = len(rows)
            source_candidates[band, i] = expected[i] >= 0 and expected[i] in rows
            for metric, values in enumerate(scores):
                row = _winner(values, rows, ids)
                predictions[band, metric, i] = row
                winning_scores[band, metric, i] = values[row] if row >= 0 else 0
    sweep, curves = [], []
    for band, (b, r) in enumerate(bands):
        for metric, name in enumerate(("jaccard", "containment")):
            for threshold in thresholds:
                predicted = np.where(winning_scores[band, metric] >= threshold, predictions[band, metric], -1)
                exact = np.where(exact_scores[metric] >= threshold, exact_predictions[metric], -1)
                metrics = retrieval_metrics(predicted, expected)
                exhaustive = retrieval_metrics(exact, expected)
                sweep.append({"b": b, "r": r, "metric": name, "threshold": threshold, **metrics,
                              "mean_candidates": float(candidate_counts[band].mean()),
                              "source_candidate_recall": float(source_candidates[band, expected >= 0].mean()),
                              "exhaustive_recall_at_1": exhaustive["recall_at_1"],
                              "relative_recall": metrics["recall_at_1"] / exhaustive["recall_at_1"] if exhaustive["recall_at_1"] else None})
        for low in np.arange(0, 1, 0.1):
            chosen = (expected >= 0) & (source_jaccard >= low) & (source_jaccard < low + 0.1 + (1e-6 if low > 0.89 else 0))
            if chosen.any():
                curves.append({"b": b, "r": r, "jaccard_low": round(float(low), 1), "pairs": int(chosen.sum()),
                               "observed_candidate_probability": float(source_candidates[band, chosen].mean()),
                               "predicted_candidate_probability": float(np.mean([
                                   candidate_probability(float(s), b, r) for s in source_jaccard[chosen]
                               ]))})
    return sweep, curves


def select_parameters(sweep: list[dict], *, f1_tolerance: float = 0.005) -> dict:
    """Choose parameters across seeds, allowing an explicit cost/accuracy tradeoff."""
    if not 0 <= f1_tolerance <= 1 or not sweep:
        raise ValueError("provide a nonempty sweep and tolerance in [0, 1]")
    groups = defaultdict(list)
    for row in sweep:
        groups[(row["vocabulary_size"], row["b"], row["r"], row["metric"], row["threshold"])].append(row)
    candidates = []
    for key, rows in groups.items():
        candidates.append({"vocabulary_size": key[0], "b": key[1], "r": key[2], "metric": key[3], "threshold": key[4],
                           "mean_f1": float(np.mean([row["f1"] for row in rows])),
                           "mean_recall_at_1": float(np.mean([row["recall_at_1"] for row in rows])),
                           "mean_precision": float(np.mean([row["precision"] for row in rows])),
                           "mean_candidates": float(np.mean([row["mean_candidates"] for row in rows]))})
    best_f1 = max(row["mean_f1"] for row in candidates)
    eligible = [row for row in candidates if row["mean_f1"] >= best_f1 - f1_tolerance]
    return min(eligible, key=lambda row: (row["mean_candidates"], -row["mean_f1"], row["vocabulary_size"],
                                          row["b"], row["r"], row["metric"], -row["threshold"]))


def project_sets(store, vocab, ids, queries, data_identity, root):
    metadata = {"version": 1, "vocabulary": vocab.identity, "data_identity": data_identity,
                "ids": ids, "queries_sha256": _digest([q.identity for q in queries])}
    path = root / f"words-{_digest(metadata)}.npz"
    try:
        with np.load(path, allow_pickle=False) as cached:
            if json.loads(cached["metadata"].item()) != metadata:
                raise ValueError("word cache metadata mismatch")
            flat, offsets = cached["words"], cached["offsets"]
            if len(offsets) != len(ids) + len(queries) + 1 or offsets[0] != 0 or offsets[-1] != len(flat) or np.any(np.diff(offsets) < 0):
                raise ValueError("invalid word cache offsets")
            sets = [flat[a:b].copy() for a, b in zip(offsets[:-1], offsets[1:])]
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        sets = [vocab.words(store.get(i)["descriptors"]) for i in ids]
        sets.extend(vocab.words(query_features(store, q)["descriptors"]) for q in queries)
        save_arrays(path, metadata, words=np.concatenate(sets),
                    offsets=np.array([0, *np.cumsum([len(s) for s in sets])], np.int64))
    return sets[:len(ids)], sets[len(ids):]


def evaluate_selected(store, vocab, ids, sets, queries, query_sets, selected, seed, *, rerank_depth=10):
    """Run raw and spatial LSH/exhaustive modes and tune spatial acceptance.

    Both modes geometrically verify the top rerank_depth set-scored candidates.
    The exhaustive mode exhausts word scores, not all descriptor comparisons.
    """
    hasher = MinHasher(len(vocab.centers), seed=seed)
    index = VisualWordIndex(sets, ids, hasher, b=selected["b"], r=selected["r"])
    lookup = {image_id: row for row, image_id in enumerate(ids)}
    expected = np.array([lookup[q.true_id] if q.true_id is not None else -1 for q in queries], np.intp)
    get_reference = lru_cache(maxsize=128)(store.get)
    # Geometry is estimated once per pair; acceptance thresholds can then be swept.
    base_spatial = SpatialConfig(min_inliers=4, min_inlier_ratio=0.0, seed=seed)
    records = []
    for i, (query, words) in enumerate(zip(queries, query_sets)):
        features = query_features(store, query)
        pair_checks = {}
        for exhaustive in (False, True):
            start = perf_counter()
            rows, scores, candidates = index.ranked_rows(words, metric=selected["metric"],
                                                         threshold=min(THRESHOLDS), exhaustive=exhaustive)
            raw_seconds = perf_counter() - start
            checks = []
            spatial_start = perf_counter()
            for row in rows[:rerank_depth]:
                if row not in pair_checks:
                    pair_checks[row] = verify_spatial(features, get_reference(ids[row]), base_spatial)
                check = pair_checks[row]
                checks.append((int(row), float(scores[row]), check))
            spatial_seconds = perf_counter() - spatial_start
            source_score = float(index.scores(words, metric=selected["metric"], rows=np.array([expected[i]]))[0]) if expected[i] >= 0 else None
            kept = rows[:rerank_depth].copy()
            records.append({"query": i, "mode": "exhaustive" if exhaustive else "lsh", "rows": kept,
                            "scores": scores[kept].copy(), "candidates": len(candidates),
                            "true_is_candidate": bool(expected[i] >= 0 and expected[i] in candidates),
                            "true_similarity": source_score, "raw_seconds": raw_seconds,
                            "spatial_seconds": spatial_seconds, "checks": checks})
        if (i + 1) % 500 == 0:
            print(f"Spatial verification seed {seed}: {i+1}/{len(queries)} queries", flush=True)
    get_reference.cache_clear()
    return records, expected


def select_spatial(runs):
    sweep = []
    for minimum in (4, 6, 10):
        for ratio in (0.2, 0.35):
            for threshold in THRESHOLDS:
                values = []
                for seed, (records, expected) in runs.items():
                    predicted = np.full(len(expected), -1, np.intp)
                    for record in records:
                        if record["mode"] == "lsh":
                            matches = _spatial_matches(record, minimum, ratio, threshold)
                            predicted[record["query"]] = matches[0][0] if matches else -1
                    values.append(retrieval_metrics(predicted, expected))
                sweep.append({"min_inliers": minimum, "min_inlier_ratio": ratio, "threshold": threshold,
                              "mean_f1": float(np.mean([v["f1"] for v in values]))})
    selected = max(sweep, key=lambda row: (row["mean_f1"], row["min_inliers"], row["min_inlier_ratio"], row["threshold"]))
    return selected, sweep


def _spatial_matches(record, minimum, ratio, threshold):
    matches = [(row, score, check) for row, score, check in record["checks"]
               if score >= threshold and check.homography is not None and check.inliers >= minimum and check.inlier_ratio >= ratio]
    # Catalogue IDs are sorted in the CLI; row order gives the same deterministic tie break.
    return sorted(matches, key=lambda m: (-m[2].inliers, -m[2].inlier_ratio, -m[1], m[0]))


def write_query_results(path, runs, ids, queries, selected, spatial):
    path.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for seed, (records, expected) in runs.items():
            predictions = {(mode, verification): np.full(len(queries), -1, np.intp)
                           for mode in ("lsh", "exhaustive") for verification in ("words", "ransac")}
            for record in records:
                query = queries[record["query"]]
                for verification in ("words", "ransac"):
                    if verification == "words":
                        row = int(record["rows"][0]) if len(record["rows"]) and record["scores"][0] >= selected["threshold"] else -1
                        score = float(record["scores"][0]) if row >= 0 else 0.0
                        inliers = 0
                    else:
                        matches = _spatial_matches(record, spatial["min_inliers"], spatial["min_inlier_ratio"], spatial["threshold"])
                        row, score, check = matches[0] if matches else (-1, 0.0, None)
                        inliers = check.inliers if check is not None else 0
                    predictions[(record["mode"], verification)][record["query"]] = row
                    writer.writerow({"system": "B", "mode": record["mode"], "kind": query.kind,
                                     "strength": query.strength, "true_id": query.true_id or "",
                                     "predicted": ids[row] if row >= 0 else "", "score": score,
                                     "candidates": record["candidates"], "true_is_candidate": record["true_is_candidate"],
                                     "true_similarity": record["true_similarity"] if record["true_similarity"] is not None else "",
                                     "lsh_seconds": record["raw_seconds"] + (record["spatial_seconds"] if verification == "ransac" else 0),
                                     "verification": verification, "source_id": query.source_id, "seed": seed,
                                     "vocabulary_size": selected["vocabulary_size"], "metric": selected["metric"], "inliers": inliers})
            for (mode, verification), predicted in predictions.items():
                by_transform = {}
                for kind in sorted({q.kind for q in queries}):
                    mask = np.array([q.kind == kind for q in queries])
                    by_transform[kind] = retrieval_metrics(predicted[mask], expected[mask])
                summaries.append({"seed": seed, "mode": mode, "verification": verification,
                                  **retrieval_metrics(predicted, expected), "by_transform": by_transform})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--limit", type=int, help="smoke cap per catalogue/fitting pool; half as many sources per validation split")
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2807, 2808])
    parser.add_argument("--sample-descriptors", type=int, default=2_000_000)
    parser.add_argument("--f1-tolerance", type=float, default=0.005)
    parser.add_argument("--rerank-depth", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/system_b_calibration.json"))
    args = parser.parse_args()
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2")
    if any(v < 2 for v in args.vocab_sizes) or args.sample_descriptors < max(args.vocab_sizes):
        parser.error("sample budget must cover each vocabulary size (>=2)")
    if args.rerank_depth < 1 or not 0 <= args.f1_tolerance <= 1 or any(s < 0 or s >= 2**31 for s in args.seeds):
        parser.error("invalid rerank depth, F1 tolerance or seed")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.vocab_sizes)) != len(args.vocab_sizes):
        parser.error("seeds and vocabulary sizes must be unique")
    cv2.setNumThreads(1)
    store = LocalFeatureStore(args.dataset)
    root = store.cache.root / "system_b"
    root.mkdir(parents=True, exist_ok=True)
    print("Preparing cached validation probes (test queries remain reserved).", flush=True)
    queries = prepare_validation_queries(store, limit=args.limit // 2 if args.limit else None)
    ids = load_split(args.dataset, "index")
    fitting = fit_image_ids(args.dataset)
    if args.limit is not None:
        required = {q.true_id for q in queries if q.true_id is not None}
        others = sorted(set(ids) - required)
        rng = np.random.default_rng(SPLIT_SEED)
        ids = sorted(required | set(rng.choice(others, min(max(0, args.limit-len(required)), len(others)), replace=False).tolist()))
        fitting = sorted(rng.choice(fitting, min(args.limit, len(fitting)), replace=False).tolist())
    training_identity = store.fingerprint(fitting)
    data_identity = store.fingerprint(sorted(set(ids) | {q.source_id for q in queries}))
    print(f"Catalogue: {len(ids)}; fitting images: {len(fitting)}; validation queries: {len(queries)}", flush=True)
    sweep, curves, vocabularies = [], [], []
    states = {}
    for seed in args.seeds:
        sample = sample_descriptors(fitting, store.get, maximum=args.sample_descriptors, seed=seed)
        if len(sample) < max(args.vocab_sizes):
            parser.error(f"only {len(sample)} fitting descriptors; reduce --vocab-sizes or increase --limit")
        sample_identity = hashlib.sha256(sample.astype("<f4").tobytes()).hexdigest()
        for size in args.vocab_sizes:
            request = {"version": 1, "size": size, "seed": seed, "training": training_identity,
                       "sample": sample_identity, "sklearn": sklearn.__version__}
            target = root / f"vocabulary-{_digest(request)}.npz"
            try:
                vocab = VisualVocabulary.load(target)
                if vocab.metadata.get("request") != request:
                    raise ValueError("vocabulary metadata mismatch")
            except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
                print(f"Fitting vocabulary k={size}, seed={seed}, descriptors={len(sample)}", flush=True)
                vocab = fit_vocabulary(sample, size, seed=seed, training_identity=training_identity)
                vocab.metadata["request"] = request
                vocab.save(target)
            print(f"Quantising words and tuning bands: k={size}, seed={seed}", flush=True)
            sets, query_sets = project_sets(store, vocab, ids, queries, data_identity, root)
            hasher = MinHasher(size, seed=seed)
            rows, observed_curves = sweep_words(sets, query_sets, ids, queries, hasher)
            sweep.extend({"seed": seed, "vocabulary_size": size, **row} for row in rows)
            curves.extend({"seed": seed, "vocabulary_size": size, **row} for row in observed_curves)
            vocabularies.append({"identity": vocab.identity, "minhash_identity": hasher.identity, **vocab.metadata})
            states[(size, seed)] = target
        del sample
    selected = select_parameters(sweep, f1_tolerance=args.f1_tolerance)
    print(f"Selected across seeds: {json.dumps(selected)}", flush=True)
    runs = {}
    for seed in args.seeds:
        vocab = VisualVocabulary.load(states[(selected["vocabulary_size"], seed)])
        sets, query_sets = project_sets(store, vocab, ids, queries, data_identity, root)
        runs[seed] = evaluate_selected(store, vocab, ids, sets, queries, query_sets, selected, seed,
                                        rerank_depth=args.rerank_depth)
    spatial, spatial_sweep = select_spatial(runs)
    csv_path = args.output.with_suffix(".csv")
    summaries = write_query_results(csv_path, runs, ids, queries, selected, spatial)
    spread = []
    for mode in ("lsh", "exhaustive"):
        for verification in ("words", "ransac"):
            group = [row for row in summaries if row["mode"] == mode and row["verification"] == verification]
            spread.append({"mode": mode, "verification": verification, **{
                metric: {"mean": float(np.mean([row[metric] for row in group])),
                         "min": min(row[metric] for row in group), "max": max(row[metric] for row in group)}
                for metric in ("recall_at_1", "precision", "f1")
            }})
    report = {
        "stage": "validation_calibration", "dataset": args.dataset, "source_limit": args.limit,
        "catalogue_size": len(ids), "fitting_images": len(fitting), "validation_queries": len(queries),
        "seeds": args.seeds, "sift": store.config.parameters, "training_identity": training_identity,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                     "opencv": cv2.__version__, "pillow": pillow_version},
        "query_identity": _digest([asdict(q) for q in queries]), "num_permutations": 128,
        "data_identity": data_identity, "source_splits": ["index", "fitting pool", "val_targets", "val_distractors"],
        "selection": f"fewest candidates within {args.f1_tolerance} of best mean validation F1 across seeds",
        "selected": selected, "spatial_selected": spatial, "spatial_sweep": spatial_sweep,
        "spatial_config": {**asdict(replace(SpatialConfig(), min_inliers=spatial["min_inliers"], min_inlier_ratio=spatial["min_inlier_ratio"])),
                           "seed": "per-run seed"},
        "rerank_depth": args.rerank_depth, "vocabularies": vocabularies, "sweep": sweep,
        "s_curves": curves, "results": summaries, "seed_spread": spread, "query_csv": str(csv_path),
        "interval_note": "Wilson intervals per transformation use distinct source images; pooled intervals ignore within-source dependence.",
        "timing_note": "Diagnostic cache-warm timings; exhaustive geometry reuses LSH pair checks. Not a speed benchmark.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved validation summary to {args.output} and per-query rows to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
