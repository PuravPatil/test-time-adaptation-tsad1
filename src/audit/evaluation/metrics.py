"""One metrics module for every method. Any change here needs a test.

- point_f1: plain point-wise F1, no adjustment
- auprc:    area under the precision-recall curve from raw scores (threshold-free)
- pa_k:     PA%K (Kim et al., AAAI 2022). A labelled anomaly segment counts as
            fully detected only if more than K% of its points were flagged.
            K = 0 is ordinary point adjustment (F1-PA); K = 100 is plain F1.
- f1_pa:    point-adjusted F1, reported only to compare with the papers
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def _check(labels, other) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(int).ravel()
    other = np.asarray(other).ravel()
    if labels.shape != other.shape:
        raise ValueError(f"length mismatch: labels {labels.shape} vs {other.shape}")
    return labels, other


def anomaly_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs for each run of 1s in labels."""
    labels = np.asarray(labels).astype(int).ravel()
    padded = np.concatenate([[0], labels, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def point_f1(labels, preds) -> float:
    labels, preds = _check(labels, preds)
    return float(f1_score(labels, preds.astype(int), zero_division=0))


def auprc(labels, scores) -> float:
    labels, scores = _check(labels, scores)
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, scores.astype(float)))


def adjust_predictions(labels, preds, k: float) -> np.ndarray:
    """PA%K adjustment of predictions; k is a percentage in [0, 100]."""
    if not 0 <= k <= 100:
        raise ValueError("k must be between 0 and 100")
    labels, preds = _check(labels, preds)
    adjusted = preds.astype(int).copy()
    for start, end in anomaly_segments(labels):
        ratio = adjusted[start:end].mean()
        if ratio > k / 100:
            adjusted[start:end] = 1
    return adjusted


def pa_k(labels, preds, k: float) -> float:
    return point_f1(labels, adjust_predictions(labels, preds, k))


def f1_pa(labels, preds) -> float:
    return pa_k(labels, preds, 0)


def evaluate(labels, scores, preds, pa_k_values=(0, 20, 50, 80, 100)) -> dict:
    """Every metric the harness reports, from one set of scores and predictions."""
    out = {
        "f1": point_f1(labels, preds),
        "auprc": auprc(labels, scores),
        "f1_pa": f1_pa(labels, preds),
    }
    for k in pa_k_values:
        out[f"pa_k_{int(k)}"] = pa_k(labels, preds, k)
    return out
