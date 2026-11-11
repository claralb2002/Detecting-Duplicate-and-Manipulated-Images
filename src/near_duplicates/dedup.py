"""Catalogue deduplication by graph clustering: plant copies, self-join with LSH, cluster the similarity graph.

    uv run python -m near_duplicates.dedup disc21 --index-size 10000 --groups 1000 --copies 3

`--groups` indexed images each get `--copies` transformed versions added to the catalogue, so a planted
group is a source with its copies and every other image is a singleton. An LSH self-join proposes pairs and
keeps those above a similarity threshold as weighted edges; the union graph joins both systems' edges.
Visual-word pairs above `--verify-jaccard` are also verified with RANSAC, giving a graph whose edges are
pairs with enough inliers, alone and joined with pHash's edges.
The graph is clustered by connected components, the baseline, and by Louvain at several resolutions, and
each partition is scored against the planted groups with the adjusted Rand index, pairwise precision and
recall, and its modularity. Sweeping the threshold shows where connected components chain groups together
through single false edges, and how far Louvain resists. Modularity's resolution limit makes Louvain at
resolution 1 merge groups as small as these, so higher resolutions are part of the sweep.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score

from near_duplicates.datasets import REPO_ROOT, load_split
from near_duplicates.features import cache_dir, Job, cached_features, init_worker
from near_duplicates.run import index_subset
from near_duplicates.systems import PhashSystem, VisualWordSystem
from near_duplicates.spatial import pair_inliers
from near_duplicates.vocabulary import fitted_vocabulary

# Realistic re-upload edits that at least one of the two systems survives at the validation thresholds.
COPY_CONDITIONS = (("jpeg", 30), ("downscale", 0.5), ("crop", 0.75), ("rotate", 5), ("overlay", 0.15),
                   ("brightness", 0.5), ("crop+jpeg+overlay", 1), ("flip", 1))


def plant(index_ids: list[str], groups: int, copies: int, dataset: str, seed: int) -> tuple[list[Job], np.ndarray]:
    """Copy jobs for `groups` random sources, and the planted group label of every catalogue item.

    The catalogue is the index followed by the copies; singletons get labels of their own.
    """
    rng = np.random.default_rng(seed)
    sources = np.sort(rng.choice(len(index_ids), size=groups, replace=False))
    jobs, labels = [], list(range(len(index_ids)))
    for source in sources:
        for c in rng.choice(len(COPY_CONDITIONS), size=copies, replace=False):
            kind, strength = COPY_CONDITIONS[c]
            jobs.append((dataset, index_ids[source], kind, strength))
            labels.append(int(source))
    return jobs, np.array(labels)


def self_join(system: PhashSystem | VisualWordSystem, encoded, similarity_of, floor: float) -> pd.DataFrame:
    """Every pair (i < j) that shares an LSH band and has similarity in [0, 1] of at least `floor`."""
    keys = system.keys(encoded)
    i_parts, j_parts, similarity_parts = [], [], []
    for i in range(len(keys)):
        candidates = system.index.candidates(keys[i])
        candidates = candidates[candidates > i]
        similarity = similarity_of(system.scores(encoded, i, candidates))
        kept = similarity >= floor
        i_parts.append(np.full(kept.sum(), i))
        j_parts.append(candidates[kept])
        similarity_parts.append(similarity[kept])
    return pd.DataFrame({"i": np.concatenate(i_parts), "j": np.concatenate(j_parts),
                         "similarity": np.concatenate(similarity_parts)})


def pairs_together(*labelings: np.ndarray) -> int:
    """Number of item pairs that share a label in every one of `labelings`."""
    _, counts = np.unique(np.stack(labelings), axis=1, return_counts=True)
    return int((counts * (counts - 1) // 2).sum())


def pair_scores(labels: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    """Precision and recall over pairs: of the pairs put together, how many are planted together, and back."""
    both, predicted, planted = pairs_together(labels, clusters), pairs_together(clusters), pairs_together(labels)
    return (both / predicted if predicted else 1.0), (both / planted if planted else 1.0)


def partition(clusters: np.ndarray) -> list[set[int]]:
    order = np.argsort(clusters, kind="stable")
    boundaries = np.flatnonzero(np.diff(clusters[order])) + 1
    return [set(part.tolist()) for part in np.split(order, boundaries)]


def evaluate(kept: pd.DataFrame, labels: np.ndarray, resolutions: list[float], seed: int) -> list[dict]:
    """Cluster the graph of `kept` edges by connected components and by Louvain at each resolution, and score it."""
    n = len(labels)
    adjacency = sp.coo_matrix((np.ones(len(kept)), (kept.i, kept.j)), shape=(n, n))
    partitions = {"connected components": connected_components(adjacency, directed=False)[1]}
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(kept[["i", "j", "similarity"]].itertuples(index=False, name=None))
    for resolution in resolutions:
        louvain = np.empty(n, dtype=np.int64)
        for c, members in enumerate(nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=seed)):
            louvain[list(members)] = c
        partitions[f"louvain r={resolution:g}"] = louvain
    rows = []
    for method, clusters in partitions.items():
        precision, recall = pair_scores(labels, clusters)
        rows.append(dict(method=method, edges=len(kept), clusters=len(np.unique(clusters)),
                         largest=int(np.bincount(clusters).max()), ari=adjusted_rand_score(labels, clusters),
                         pair_precision=precision, pair_recall=recall,
                         modularity=nx.community.modularity(graph, partition(clusters)) if len(kept) else 0.0))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=("caltech101", "disc21"))
    parser.add_argument("--index-size", type=int)
    parser.add_argument("--groups", type=int, default=1000)
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument("--vocabulary", type=int, default=2048)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--minhash-bands", type=int, default=64)
    parser.add_argument("--minhash-rows", type=int, default=2)
    parser.add_argument("--bit-bands", type=int, default=16)
    parser.add_argument("--bit-rows", type=int, default=8)
    parser.add_argument("--hamming", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 14, 16])
    parser.add_argument("--jaccard", type=float, nargs="+", default=[0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1])
    parser.add_argument("--union", nargs="+", default=["8:0.3", "10:0.3", "12:0.25", "14:0.2"],
                        help="HAMMING:JACCARD pairs for the union graph, an edge if either system passes")
    parser.add_argument("--resolution", type=float, nargs="+", default=[1, 10, 100], help="Louvain resolutions")
    parser.add_argument("--verify-jaccard", type=float, default=0.1, help="visual-word pairs above this are RANSAC-verified")
    parser.add_argument("--inliers", type=int, nargs="+", default=[15, 20, 25, 30, 40], help="inlier thresholds for verified edges")
    parser.add_argument("--verified-union", type=int, nargs="+", default=[8, 10], help="pHash distances joined with verified edges")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_index = load_split(args.dataset, "index")
    positions = index_subset(full_index, set(), args.index_size, args.seed)
    index_ids = [full_index[i] for i in positions]
    jobs, labels = plant(index_ids, args.groups, args.copies, args.dataset, args.seed)
    cache = cache_dir(args.dataset)
    with Pool(args.workers, initializer=init_worker) as pool:
        full = cached_features(cache / f"index_n{len(full_index)}.npz",
                               [(args.dataset, i, "clean", 0) for i in full_index], pool, "index")
        vocabulary = fitted_vocabulary(args.dataset, full.ids, full.descriptors, args.vocabulary, args.seed)
        copies = cached_features(cache / "dedup" / f"copies_n{len(index_ids)}_g{args.groups}_c{args.copies}_s{args.seed}.npz",
                                 jobs, pool, "copies")
    index = full.subset(positions)
    copies.ids = [f"copy{n}:{j[1]}" for n, j in enumerate(jobs)]
    catalogue = index.concatenate(copies)
    print(f"{len(catalogue.ids):,} catalogue images, {args.groups} planted groups of {args.copies + 1}", file=sys.stderr)

    phash = PhashSystem(catalogue.hashes, args.bit_bands, args.bit_rows, seed=args.seed)
    words = VisualWordSystem(catalogue.descriptors, vocabulary, args.num_perm, args.minhash_bands, args.minhash_rows, seed=args.seed)
    union = [tuple(map(float, pair.split(":"))) for pair in args.union]
    phash_edges = self_join(phash, catalogue.hashes, lambda s: 1 + s / 64, 1 - max(args.hamming + [h for h, _ in union]) / 64)
    word_edges = self_join(words, words.encode(catalogue), lambda s: s, min(args.jaccard + [j for _, j in union]))
    print(f"self-join pairs above the loosest threshold: {len(phash_edges):,} pHash, {len(word_edges):,} visual words",
          file=sys.stderr)
    graphs = [("phash", f"d<={h}", phash_edges[phash_edges.similarity >= 1 - h / 64]) for h in args.hamming]
    graphs += [("visual-words", f"J>={j:g}", word_edges[word_edges.similarity >= j]) for j in args.jaccard]
    for h, j in union:
        # The two similarities have different scales, so union edges are unweighted.
        either = pd.concat([phash_edges[phash_edges.similarity >= 1 - h / 64], word_edges[word_edges.similarity >= j]])
        graphs.append(("union", f"d<={h:g} or J>={j:g}", either.drop_duplicates(["i", "j"]).assign(similarity=1.0)))
    to_verify = word_edges[word_edges.similarity >= args.verify_jaccard]
    started = time.perf_counter()
    counts = pair_inliers(to_verify.i.to_numpy(), to_verify.j.to_numpy(), catalogue.keypoints, catalogue.descriptors, args.workers)
    print(f"RANSAC on {len(to_verify):,} pairs: {time.perf_counter() - started:.0f} s", file=sys.stderr)
    verified = to_verify.assign(similarity=counts.astype(float))
    for t in args.inliers:
        graphs.append(("verified", f"inliers>={t}", verified[verified.similarity >= t]))
        for h in args.verified_union:
            either = pd.concat([phash_edges[phash_edges.similarity >= 1 - h / 64], verified[verified.similarity >= t]])
            graphs.append(("union verified", f"d<={h} or inliers>={t}", either.drop_duplicates(["i", "j"]).assign(similarity=1.0)))
    rows = [dict(system=system, threshold=threshold, **row)
            for system, threshold, kept in graphs for row in evaluate(kept, labels, args.resolution, args.seed)]
    table = pd.DataFrame(rows)
    out = REPO_ROOT / "results" / f"dedup_{args.dataset}_n{len(index_ids)}_g{args.groups}_c{args.copies}_s{args.seed}.csv"
    table.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
