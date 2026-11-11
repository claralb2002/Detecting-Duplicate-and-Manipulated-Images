"""Spatial verification: count the descriptor matches consistent with one affine map between two images.

A shared visual word says two images contain similar patches; it does not say the patches sit in the same
arrangement. Matching descriptors (Lowe's ratio test) and fitting an affine map with RANSAC keeps only
matches that agree geometrically, so a true near-duplicate keeps many inliers and an unrelated image few.
The full affine model (6 degrees of freedom) covers scaling, rotation, cropping and also mirroring.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

RATIO = 0.8
REPROJECTION_PIXELS = 8.0
MIN_MATCHES = 4


def ratio_matches(query: np.ndarray, reference: np.ndarray, ratio: float = RATIO) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of query and reference descriptors whose nearest neighbour is clearly closer than the second."""
    if len(query) < 1 or len(reference) < 2:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query.astype(np.float32), reference.astype(np.float32), k=2)
    kept = [(best.queryIdx, best.trainIdx) for best, second in pairs if best.distance < ratio * second.distance]
    if not kept:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    query_rows, reference_rows = np.array(kept).T
    return query_rows, reference_rows


def inliers(query_points: np.ndarray, reference_points: np.ndarray, query_descriptors: np.ndarray,
            reference_descriptors: np.ndarray) -> int:
    """The number of ratio-test matches that one RANSAC affine map from query to reference explains.

    The RANSAC seed is reset for every pair, so the count does not depend on the order of the calls.
    """
    query_rows, reference_rows = ratio_matches(query_descriptors, reference_descriptors)
    if len(query_rows) < MIN_MATCHES:
        return 0
    cv2.setRNGSeed(0)
    _, mask = cv2.estimateAffine2D(query_points[query_rows], reference_points[reference_rows], method=cv2.RANSAC,
                                   ransacReprojThreshold=REPROJECTION_PIXELS, maxIters=2000, confidence=0.99)
    return 0 if mask is None else int(mask.sum())


def pair_inliers(i: np.ndarray, j: np.ndarray, keypoints: list[np.ndarray], descriptors: list[np.ndarray],
                 workers: int) -> np.ndarray:
    """`inliers` for every pair (i[n], j[n]) of images in one collection, computed on `workers` threads.

    OpenCV releases the GIL inside matching and RANSAC, and its random generator is per thread, so threads
    share the descriptors without copying them and every count equals a sequential call's.
    """
    def count(pair: tuple[int, int]) -> int:
        a, b = pair
        return inliers(keypoints[a], keypoints[b], descriptors[a], descriptors[b])

    with ThreadPoolExecutor(workers) as pool:
        return np.fromiter(pool.map(count, zip(i.tolist(), j.tolist())), dtype=np.int64, count=len(i))
