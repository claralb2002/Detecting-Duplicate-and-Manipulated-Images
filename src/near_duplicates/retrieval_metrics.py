"""Retrieval metrics with explicit abstentions, wrong-source errors and intervals."""

from math import sqrt

import numpy as np


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if not 0 <= successes <= total:
        raise ValueError("successes must lie between zero and total")
    if not total:
        return None
    z = 1.959963984540054
    p, denominator = successes / total, 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def retrieval_metrics(predicted: np.ndarray, expected: np.ndarray) -> dict:
    predicted, expected = np.asarray(predicted), np.asarray(expected)
    if predicted.shape != expected.shape or predicted.ndim != 1:
        raise ValueError("predictions and expected rows must have the same vector shape")
    positive, accepted = expected >= 0, predicted >= 0
    tp = int((accepted & positive & (predicted == expected)).sum())
    fp, fn = int(accepted.sum()) - tp, int(positive.sum()) - tp
    return {
        "queries": len(expected), "positive_queries": int(positive.sum()), "negative_queries": int((~positive).sum()),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "recall_at_1": tp / int(positive.sum()) if positive.any() else 0.0,
        "precision": tp / (tp+fp) if tp+fp else 0.0,
        "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
        "distractor_false_positive_rate": float((accepted & ~positive).sum()) / int((~positive).sum()) if (~positive).any() else 0.0,
        "recall_wilson95": wilson_interval(tp, int(positive.sum())),
    }
