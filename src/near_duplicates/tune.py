"""Choose LSH settings, the RANSAC shortlist and the vocabulary size from one validation run, without new searches.

    uv run python -m near_duplicates.tune results/queries_disc21_n50000_k2048_p256_mb85r3_bb24r12_t10_s0_val.csv --vocabulary 512 2048 8192

The s-curve predicts LSH recall from each target's true similarity (docs/findings.md, 1), so every setting can be
scored at once. For every validation query the similarity to every indexed image is computed once: the
Hamming distance for pHash, and the visual-word Jaccard similarity for each vocabulary size. From those:

- recall at the false-match threshold with LSH is predicted as [accepted by exhaustive search] x
  P(candidate | true similarity);
- the candidate fraction is predicted as the mean of P(candidate | similarity) over all query-image pairs;
- spatial verification accepts a target if its true image enters the top-k shortlist by Jaccard similarity
  and has more RANSAC inliers than the threshold.

RANSAC inliers do not depend on the vocabulary, so the inliers of each true pair measured by `run` are reused
for every vocabulary and shortlist; the inlier threshold is the run's, so for shortlists longer than the run's
the prediction ignores the extra false matches a longer shortlist admits. The run's own settings are
predicted too, against what it measured, so the error of the method is printed beside its choices.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from near_duplicates.datasets import load_split
from near_duplicates.features import cache_dir, cached_features, load_features, worker_pool
from near_duplicates.lsh import bit_sampling_curve, s_curve
from near_duplicates.metrics import is_target, threshold_at_fmr
from near_duplicates.phash import HASH_BITS
from near_duplicates.run import index_subset
from near_duplicates.transforms import CONDITIONS
from near_duplicates.vocabulary import fitted_vocabulary, incidence, word_sets

JACCARD_BINS = 10_000
VERIFIED = "visual-words+ransac"
RUN_TAG = re.compile(r"queries_(?P<dataset>[a-z0-9]+)_n(?P<index_size>\d+)_k(?P<vocabulary>\d+)_p(?P<num_perm>\d+)"
                     r"_mb(?P<minhash_bands>\d+)r(?P<minhash_rows>\d+)_bb(?P<bit_bands>\d+)r(?P<bit_rows>\d+)"
                     r"_t(?P<top_k>\d+)_s(?P<seed>\d+)_val$")


@dataclass
class Profile:
    """One representation's view of every query against the whole index.

    `rank` counts indexed images strictly more similar than the true one (0 if it is the best; -1 for
    distractors). `histogram[i]` counts query-image pairs whose similarity falls in bin i of `values`.
    """

    true_similarity: np.ndarray
    rank: np.ndarray
    top_correct: np.ndarray
    top_score: np.ndarray
    histogram: np.ndarray
    values: np.ndarray

    def candidate_fraction(self, curve) -> float:
        return float((self.histogram * curve(self.values)).sum() / self.histogram.sum())


def summarize_rows(similarity: np.ndarray, true_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """True similarity, rank of the true image, whether the most similar image is the true one, and its score."""
    rows = np.arange(len(similarity))
    target = true_positions >= 0
    true_similarity = np.where(target, similarity[rows, np.maximum(true_positions, 0)], np.nan)
    rank = np.where(target, (similarity > true_similarity[:, None]).sum(axis=1), -1)
    best = similarity.argmax(axis=1)
    return true_similarity, rank, target & (best == true_positions), similarity[rows, best]


def jaccard_bins(similarity: np.ndarray) -> np.ndarray:
    """Bin 0 holds exact zeros, which LSH never proposes; bins 1.. cover (0, 1] in steps of 1 / JACCARD_BINS."""
    return np.where(similarity > 0, 1 + np.minimum(similarity * JACCARD_BINS, JACCARD_BINS - 1).astype(np.int64), 0)


JACCARD_VALUES = np.concatenate([[0.0], (np.arange(JACCARD_BINS) + 0.5) / JACCARD_BINS])


def jaccard_profile(index: np.ndarray, index_sizes: np.ndarray, queries: np.ndarray, true_positions: np.ndarray) -> Profile:
    """Profile of binary word-incidence rows `queries` against the dense incidence matrix `index` with row sums `index_sizes`."""
    intersection = queries @ index.T
    union = queries.sum(axis=1)[:, None] + index_sizes[None, :] - intersection
    similarity = intersection / np.maximum(union, 1)
    histogram = np.bincount(jaccard_bins(similarity).ravel(), minlength=JACCARD_BINS + 1)
    return Profile(*summarize_rows(similarity, true_positions), histogram, JACCARD_VALUES)


def hamming_profile(index: np.ndarray, queries: np.ndarray, true_positions: np.ndarray) -> Profile:
    """Profile of 64-bit codes; the similarity is the negated Hamming distance, as pHash scores it."""
    distance = np.bitwise_count(index[None, :] ^ queries[:, None]).astype(np.int64)
    histogram = np.bincount(distance.ravel(), minlength=HASH_BITS + 1)
    return Profile(*summarize_rows(-distance, true_positions), histogram, np.arange(HASH_BITS + 1))


def concatenate(parts: list[Profile]) -> Profile:
    return Profile(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("true_similarity", "rank", "top_correct", "top_score")),
                   sum(p.histogram for p in parts), parts[0].values)


def accepted_at(profile: Profile, target: np.ndarray, fmr: float) -> np.ndarray:
    """Targets that exhaustive search answers correctly above the threshold set on the distractors."""
    threshold = threshold_at_fmr(profile.top_score[~target], fmr)
    return target & profile.top_correct & (profile.top_score > threshold)


def bit_sampling_grid(profile: Profile, accepted: np.ndarray, target: np.ndarray) -> pd.DataFrame:
    distance = np.rint(-profile.true_similarity[target]).astype(int)
    rows = []
    for r in range(4, 17):
        for b in range(1, 65):
            curve = lambda d: bit_sampling_curve(d, b, r)
            rows.append(dict(system="phash", bands=b, rows=r, predicted_recall=float((accepted[target] * curve(distance)).mean()),
                             candidate_fraction=profile.candidate_fraction(curve)))
    return pd.DataFrame(rows).assign(exhaustive_recall=accepted[target].mean())


def minhash_grid(profile: Profile, system: str, accepted: np.ndarray, target: np.ndarray, num_perms: list[int]) -> pd.DataFrame:
    jaccard = profile.true_similarity[target]
    rows = []
    for num_perm in num_perms:
        for r in range(1, 9):
            for b in range(1, num_perm // r + 1):
                curve = lambda s: s_curve(s, b, r)
                rows.append(dict(system=system, num_perm=num_perm, bands=b, rows=r,
                                 predicted_recall=float((accepted[target] * curve(jaccard)).mean()),
                                 candidate_fraction=profile.candidate_fraction(curve)))
    return pd.DataFrame(rows).assign(exhaustive_recall=accepted[target].mean())


def cheapest(grid: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Per group, the setting with the fewest candidates whose predicted recall is within `tolerance` of exhaustive."""
    keys = [c for c in ("system", "vocabulary", "top_k") if c in grid]
    good = grid[grid.predicted_recall >= grid.exhaustive_recall - tolerance]
    return good.sort_values(["candidate_fraction", "bands"]).groupby(keys, dropna=False).head(1).sort_values(keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queries", type=Path, help="a full validation results/queries_<tag>_val.csv")
    parser.add_argument("--vocabulary", type=int, nargs="+", default=[2048])
    parser.add_argument("--num-perm", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 25, 50])
    parser.add_argument("--tolerance", type=float, default=0.005, help="recall LSH may lose against exhaustive search")
    parser.add_argument("--target-fmr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    match = RUN_TAG.match(args.queries.stem)
    if match is None:
        parser.error("expected a full validation file named by `run`, e.g. queries_<dataset>_n..._t10_s0_val.csv")
    run = {k: (v if k == "dataset" else int(v)) for k, v in match.groupdict().items()}
    results = pd.read_csv(args.queries, keep_default_na=False, na_values={"true_similarity": [""]})
    results["strength"] = results.strength.astype(float)

    dataset, seed = run["dataset"], run["seed"]
    full_index = load_split(dataset, "index")
    targets, distractors = load_split(dataset, "val_targets"), load_split(dataset, "val_distractors")
    positions = index_subset(full_index, set(targets), run["index_size"], seed)
    cache = cache_dir(dataset)
    with worker_pool(args.workers) as pool:
        full = cached_features(cache / f"index_n{len(full_index)}.npz", [(dataset, i, "clean", 0) for i in full_index], pool, "index")
        vocabularies = {k: fitted_vocabulary(dataset, full.ids, full.descriptors, k, seed) for k in args.vocabulary}
    index = full.subset(positions)
    position_of = {image_id: i for i, image_id in enumerate(index.ids)}
    dense = {k: incidence(word_sets(index.descriptors, vocabulary), k).toarray() for k, vocabulary in vocabularies.items()}
    sizes = {k: matrix.sum(axis=1) for k, matrix in dense.items()}

    ids = targets + distractors
    true_positions = np.array([position_of[t] for t in targets] + [-1] * len(distractors))
    hamming_parts, jaccard_parts, keys = [], {k: [] for k in vocabularies}, []
    for kind, strength in CONDITIONS:
        queries = load_features(cache / "queries" / f"val_{kind}_{strength}_n{len(ids)}.npz", ids)
        if queries is None:
            sys.exit(f"query features for {kind} {strength} are missing: run the validation split first")
        print(f"{kind} {strength}", file=sys.stderr)
        hamming_parts.append(hamming_profile(index.hashes, queries.hashes, true_positions))
        for k, vocabulary in vocabularies.items():
            jaccard_parts[k].append(jaccard_profile(dense[k], sizes[k], incidence(word_sets(queries.descriptors, vocabulary), k).toarray(),
                                                    true_positions))
        keys += [(kind, float(strength), q) for q in ids]
    target = np.tile(true_positions >= 0, len(CONDITIONS))
    hamming = concatenate(hamming_parts)
    jaccard = {k: concatenate(parts) for k, parts in jaccard_parts.items()}

    verified = results[(results.system == VERIFIED) & (results["mode"] == "exhaustive")]
    inlier_threshold = threshold_at_fmr(verified.score[~is_target(verified)].to_numpy(), args.target_fmr)
    inliers = verified.set_index(["kind", "strength", "query_id"]).true_similarity.reindex(keys).to_numpy()
    measured_verified = ((verified.predicted == verified.true_id) & (verified.score > inlier_threshold))[is_target(verified)].mean()

    grids = [bit_sampling_grid(hamming, accepted_at(hamming, target, args.target_fmr), target)]
    study = []
    for k, profile in jaccard.items():
        accepted = accepted_at(profile, target, args.target_fmr)
        grids.append(minhash_grid(profile, "visual-words", accepted, target, args.num_perm).assign(vocabulary=k))
        for top_k in args.top_k:
            shortlisted = target & (profile.rank >= 0) & (profile.rank < top_k) & (inliers > inlier_threshold)
            grids.append(minhash_grid(profile, VERIFIED, shortlisted, target, args.num_perm).assign(vocabulary=k, top_k=top_k))
        study.append(dict(vocabulary=k, davies_bouldin=vocabularies[k].metadata["davies_bouldin"],
                          mean_true_jaccard=np.nanmean(profile.true_similarity), recall_at_1=profile.top_correct[target].mean(),
                          recall_at_threshold=accepted[target].mean(),
                          **{f"verified_top{t}": (target & (profile.rank >= 0) & (profile.rank < t) & (inliers > inlier_threshold))[target].mean()
                             for t in args.top_k}))
    grid = pd.concat(grids, ignore_index=True)
    out = args.queries.parent / f"tuning_{args.queries.stem.removeprefix('queries_')}.csv"
    grid.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print("\nThe method checked against the run's own settings")
    measured = lambda system: results[(results.system == system) & (results["mode"] == "lsh")].candidates.mean() / len(index.ids)
    print(f"phash candidate fraction: measured {measured('phash'):.4f}, predicted "
          f"{hamming.candidate_fraction(lambda d: bit_sampling_curve(d, run['bit_bands'], run['bit_rows'])):.4f}")
    if (own := jaccard.get(run["vocabulary"])) is not None:
        print(f"visual-words candidate fraction: measured {measured('visual-words'):.4f}, predicted "
              f"{own.candidate_fraction(lambda s: s_curve(s, run['minhash_bands'], run['minhash_rows'])):.4f}")
        shortlisted = target & (own.rank >= 0) & (own.rank < run["top_k"]) & (inliers > inlier_threshold)
        print(f"{VERIFIED}, top {run['top_k']}, exhaustive recall at the threshold: measured {measured_verified:.3f}, "
              f"predicted {shortlisted[target].mean():.3f}")
    print(f"\nVocabulary study (validation, exhaustive retrieval)\n\n{pd.DataFrame(study).round(3).to_string(index=False)}")
    print(f"\nCheapest LSH within {args.tolerance} of exhaustive recall at the threshold\n\n"
          f"{cheapest(grid, args.tolerance).round(4).to_string(index=False)}\n\nfull grid: {out}")


if __name__ == "__main__":
    main()
