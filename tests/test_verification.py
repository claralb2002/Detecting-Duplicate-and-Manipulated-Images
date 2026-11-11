from __future__ import annotations

import numpy as np

from near_duplicates.features import FeatureSet, load_image, phash, sift_features
from near_duplicates.systems import SpatiallyVerifiedSystem, VisualWordSystem
from near_duplicates.transforms import transform
from near_duplicates.verification import inliers, pair_inliers, ratio_matches
from near_duplicates.vocabulary import build_vocabulary


def features_of(images) -> FeatureSet:
    extracted = [sift_features(img) for img in images]
    return FeatureSet([f"i{n}" for n in range(len(images))], np.array([phash(img) for img in images], dtype=np.uint64),
                      [d for _, d in extracted], [k for k, _ in extracted])


def test_geometric_edits_keep_more_inliers_than_any_unrelated_image(caltech_sample):
    images = [load_image(p) for p in caltech_sample[:12]]
    reference = features_of(images)
    rng = np.random.default_rng(0)
    for kind, strength in (("crop", 0.5), ("rotate", 15), ("downscale", 0.5), ("overlay", 0.3)):
        edited = features_of([transform(img, kind, strength, rng) for img in images[:6]])
        same = [inliers(edited.keypoints[i], reference.keypoints[i], edited.descriptors[i], reference.descriptors[i])
                for i in range(6)]
        other = [inliers(edited.keypoints[i], reference.keypoints[j], edited.descriptors[i], reference.descriptors[j])
                 for i in range(6) for j in range(6, 12)]
        assert min(same) > max(other), (kind, same, max(other))


def test_inlier_count_does_not_depend_on_call_order(caltech_sample):
    f = features_of([load_image(p) for p in caltech_sample[:3]])
    q = features_of([transform(load_image(caltech_sample[0]), "rotate", 30, np.random.default_rng(0))])
    first = inliers(q.keypoints[0], f.keypoints[0], q.descriptors[0], f.descriptors[0])
    inliers(q.keypoints[0], f.keypoints[1], q.descriptors[0], f.descriptors[1])
    assert inliers(q.keypoints[0], f.keypoints[0], q.descriptors[0], f.descriptors[0]) == first


def test_images_without_descriptors_have_no_matches():
    empty = np.zeros((0, 128), np.uint8)
    some = np.random.default_rng(0).integers(0, 256, (5, 128), dtype=np.uint8)
    assert len(ratio_matches(empty, some)[0]) == 0 and len(ratio_matches(some, empty)[0]) == 0
    assert inliers(np.zeros((0, 2), np.float32), np.zeros((5, 2), np.float32), empty, some) == 0


def test_verified_system_reranks_its_shortlist_by_inliers(caltech_sample):
    images = [load_image(p) for p in caltech_sample[:30]]
    index = features_of(images)
    centers = build_vocabulary(index.descriptors, k=64, seed=0, sample_size=20_000)
    words = VisualWordSystem(index.descriptors, centers, num_perm=64, bands=32, rows=2, seed=0)
    queries = features_of([transform(images[i], "crop", 0.3, np.random.default_rng(i)) for i in (3, 17)])
    everything = SpatiallyVerifiedSystem(words, index.descriptors, index.keypoints, top_k=30)
    shortlist = SpatiallyVerifiedSystem(words, index.descriptors, index.keypoints, top_k=3)
    encoded = everything.encode(queries)
    for query, source in enumerate((3, 17)):
        # With every image shortlisted, the source wins on inliers even where Jaccard ranks it low.
        position, score = everything.rank(encoded, query)
        assert position == source and int(score) == everything.similarity(encoded, query, source)
        # A short list only ever returns one of the top Jaccard images among the allowed positions.
        allowed = np.array([p for p in range(30) if p != source])
        jaccard = words.scores(encoded[0], query, allowed)
        assert shortlist.rank(encoded, query, allowed)[0] in allowed[np.argsort(-jaccard, kind="stable")[:3]]


def test_threaded_pair_inliers_equal_sequential_counts(caltech_sample):
    rng = np.random.default_rng(0)
    features = [sift_features(transform(load_image(p), "rotate", 10, rng)) for p in caltech_sample[:12]]
    features += [sift_features(load_image(p)) for p in caltech_sample[:12]]
    keypoints, descriptors = [k for k, _ in features], [d for _, d in features]
    i, j = np.repeat(np.arange(12), 3), np.concatenate([[n + 12, (n + 1) % 12 + 12, (n + 5) % 12 + 12] for n in range(12)])
    threaded = pair_inliers(i, j, keypoints, descriptors, workers=6)
    assert threaded.tolist() == [inliers(keypoints[a], keypoints[b], descriptors[a], descriptors[b]) for a, b in zip(i, j)]
    assert (threaded[::3] > threaded[1::3]).mean() > 0.8  # each rotated image keeps more inliers with its source
