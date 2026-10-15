import numpy as np
import pytest

from audit.evaluation.metrics import (
    adjust_predictions, anomaly_segments, auprc, auroc, evaluate, f1_best, f1_pa, pa_k, point_f1,
)

# One 4-point anomaly segment (indices 2-5) and one 2-point segment (8-9).
LABELS = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
# Flags 1 of 4 points in the first segment, none in the second.
PREDS = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])


def test_segments():
    assert anomaly_segments(LABELS) == [(2, 6), (8, 10)]


def test_point_f1():
    # precision 1/1, recall 1/6
    assert point_f1(LABELS, PREDS) == pytest.approx(2 * 1 * (1 / 6) / (1 + 1 / 6))


def test_f1_pa_fills_the_whole_segment():
    adjusted = adjust_predictions(LABELS, PREDS, 0)
    assert adjusted.tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    # precision 4/4, recall 4/6
    assert f1_pa(LABELS, PREDS) == pytest.approx(0.8)


def test_pa_k_respects_k():
    # 25% of the first segment was flagged: adjusted at K=20, not at K=50
    assert pa_k(LABELS, PREDS, 20) == pytest.approx(f1_pa(LABELS, PREDS))
    assert pa_k(LABELS, PREDS, 50) == pytest.approx(point_f1(LABELS, PREDS))


def test_pa_k_100_equals_plain_f1():
    assert pa_k(LABELS, PREDS, 100) == pytest.approx(point_f1(LABELS, PREDS))


def test_auprc_perfect_scores():
    assert auprc(LABELS, LABELS.astype(float)) == pytest.approx(1.0)


def test_evaluate_keys():
    out = evaluate(LABELS, PREDS.astype(float), PREDS, pa_k_values=(0, 50))
    assert set(out) == {"f1", "auprc", "auroc", "f1_best", "f1_pa", "pa_k_0", "pa_k_50"}


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        point_f1(LABELS, PREDS[:-1])


def test_auroc_perfect_and_inverted():
    assert auroc(LABELS, LABELS.astype(float)) == pytest.approx(1.0)
    assert auroc(LABELS, 1.0 - LABELS) == pytest.approx(0.0)


def test_f1_best_perfect_scores():
    assert f1_best(LABELS, LABELS.astype(float)) == pytest.approx(1.0)
