from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from near_duplicates.dedup import COPY_CONDITIONS, evaluate, pair_scores, partition, plant, self_join
from near_duplicates.systems import PhashSystem


def test_pair_scores_match_counting_every_pair():
    rng = np.random.default_rng(0)
    labels, clusters = rng.integers(0, 6, 40), rng.integers(0, 6, 40)
    pairs = list(itertools.combinations(range(40), 2))
    planted = {(i, j) for i, j in pairs if labels[i] == labels[j]}
    predicted = {(i, j) for i, j in pairs if clusters[i] == clusters[j]}
    precision, recall = pair_scores(labels, clusters)
    assert precision == pytest.approx(len(planted & predicted) / len(predicted))
    assert recall == pytest.approx(len(planted & predicted) / len(planted))


def test_partition_groups_items_by_cluster():
    assert sorted(map(sorted, partition(np.array([2, 0, 2, 1, 0])))) == [[0, 2], [1, 4], [3]]


def test_plant_labels_each_copy_with_its_source():
    ids = [f"r{i}" for i in range(50)]
    jobs, labels = plant(ids, groups=5, copies=3, dataset="disc21", seed=0)
    assert len(jobs) == 15 and len(labels) == 65 and (labels[:50] == np.arange(50)).all()
    for job, label in zip(jobs, labels[50:]):
        assert job[1] == ids[label] and (job[2], job[3]) in COPY_CONDITIONS
    assert all(len({j[2] for j in jobs[g * 3:(g + 1) * 3]}) == 3 for g in range(5))  # distinct edits per group


def test_one_false_edge_merges_two_groups_for_connected_components_but_not_louvain():
    # Two planted 5-cliques joined by one weak false edge, plus singletons.
    labels = np.array([0] * 5 + [1] * 5 + list(range(2, 12)))
    clique = lambda nodes: [(i, j, 0.9) for i, j in itertools.combinations(nodes, 2)]
    edges = pd.DataFrame(clique(range(5)) + clique(range(5, 10)) + [(4, 5, 0.5)], columns=["i", "j", "similarity"])
    rows = {r["method"]: r for r in evaluate(edges, labels, resolutions=[1], seed=0)}
    assert rows["connected components"]["largest"] == 10 and rows["connected components"]["pair_precision"] < 0.5
    assert rows["louvain r=1"]["ari"] == pytest.approx(1.0) and rows["louvain r=1"]["pair_precision"] == 1.0
    cut = {r["method"]: r for r in evaluate(edges[edges.similarity > 0.6], labels, resolutions=[1], seed=0)}
    assert cut["connected components"]["ari"] == pytest.approx(1.0)


def test_low_resolution_louvain_merges_many_small_groups_and_high_resolution_separates_them():
    # 30 planted pairs in a ring of weak false edges: modularity at resolution 1 prefers merging neighbours.
    labels = np.repeat(np.arange(30), 2)
    edges = [(2 * g, 2 * g + 1, 1.0) for g in range(30)] + [(2 * g + 1, (2 * g + 2) % 60, 0.6) for g in range(30)]
    rows = {r["method"]: r for r in evaluate(pd.DataFrame(edges, columns=["i", "j", "similarity"]), labels, [1, 10], seed=0)}
    assert rows["connected components"]["clusters"] == 1
    assert rows["louvain r=1"]["clusters"] < 30 and rows["louvain r=10"]["ari"] == pytest.approx(1.0)


def test_self_join_returns_each_close_pair_once():
    rng = np.random.default_rng(0)
    hashes = rng.integers(0, 2**64, 200, dtype=np.uint64)
    hashes[150] = hashes[3] ^ np.uint64(0b101)  # distance 2
    system = PhashSystem(hashes, 16, 8, seed=0)
    edges = self_join(system, hashes, lambda s: 1 + s / 64, floor=1 - 4 / 64)
    assert edges[["i", "j"]].values.tolist() == [[3, 150]] and edges.similarity.item() == pytest.approx(62 / 64)
