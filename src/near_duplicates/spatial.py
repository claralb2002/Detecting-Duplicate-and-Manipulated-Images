"""SIFT ratio-test correspondences and RANSAC homography verification."""

from dataclasses import dataclass

import cv2
import numpy as np

from near_duplicates.visual_words import descriptor_array


@dataclass(frozen=True)
class SpatialConfig:
    ratio: float = 0.75
    reprojection_threshold: float = 3.0
    min_inliers: int = 6
    min_inlier_ratio: float = 0.25
    max_iterations: int = 2000
    confidence: float = 0.995
    seed: int = 2807

    def __post_init__(self):
        if not 0 < self.ratio < 1 or not 0 <= self.min_inlier_ratio <= 1 or not 0 < self.confidence < 1:
            raise ValueError("invalid spatial ratio/confidence")
        if not np.isfinite(self.reprojection_threshold) or self.reprojection_threshold <= 0:
            raise ValueError("reprojection threshold must be finite and positive")
        if type(self.min_inliers) is not int or self.min_inliers < 4:
            raise ValueError("a homography needs at least four inliers")
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")


@dataclass(frozen=True)
class SpatialResult:
    accepted: bool
    matches: int
    inliers: int
    inlier_ratio: float
    homography: np.ndarray | None = None


def verify_spatial(query: dict, reference: dict, config: SpatialConfig = SpatialConfig()) -> SpatialResult:
    qdesc, rdesc = descriptor_array(query["descriptors"]), descriptor_array(reference["descriptors"])
    qpoints, rpoints = np.asarray(query["keypoints"], dtype=np.float32), np.asarray(reference["keypoints"], dtype=np.float32)
    if qpoints.shape != (len(qdesc), 2) or rpoints.shape != (len(rdesc), 2):
        raise ValueError("keypoints must have shape (descriptors, 2)")
    if not np.isfinite(qpoints).all() or not np.isfinite(rpoints).all():
        raise ValueError("keypoints must be finite")
    if len(qdesc) < 4 or len(rdesc) < 4:
        return SpatialResult(False, 0, 0, 0.0)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(qdesc, rdesc, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < config.ratio * pair[1].distance]
    # Enforce distinct reference descriptors; repeated textures cannot vote many times.
    unique, query_locations, reference_locations = {}, set(), set()
    for match in sorted(good, key=lambda m: (m.distance, m.queryIdx, m.trainIdx)):
        qlocation = tuple(qpoints[match.queryIdx])
        rlocation = tuple(rpoints[match.trainIdx])
        if match.trainIdx in unique or qlocation in query_locations or rlocation in reference_locations:
            continue
        unique[match.trainIdx] = match
        query_locations.add(qlocation)
        reference_locations.add(rlocation)
    good = list(unique.values())
    if len(good) < 4:
        return SpatialResult(False, len(good), 0, 0.0)
    src = np.array([qpoints[m.queryIdx] for m in good], np.float32)
    dst = np.array([rpoints[m.trainIdx] for m in good], np.float32)
    # Reject collinear geometry even when descriptor matches are perfect.
    if np.linalg.matrix_rank(src - src.mean(axis=0)) < 2 or np.linalg.matrix_rank(dst - dst.mean(axis=0)) < 2:
        return SpatialResult(False, len(good), 0, 0.0)
    cv2.setRNGSeed(config.seed)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, config.reprojection_threshold,
                                     maxIters=config.max_iterations, confidence=config.confidence)
    if matrix is None or mask is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        return SpatialResult(False, len(good), 0, 0.0)
    inliers = int(mask.sum())
    ratio = inliers / len(good)
    return SpatialResult(inliers >= config.min_inliers and ratio >= config.min_inlier_ratio,
                          len(good), inliers, ratio, matrix)
