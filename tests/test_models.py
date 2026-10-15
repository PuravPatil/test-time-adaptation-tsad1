"""End-to-end smoke tests for the two detectors on a synthetic stream (needs torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from audit.methods.candi_adapter import CANDIAdapter  # noqa: E402
from audit.methods.m2n2_adapter import M2N2Adapter    # noqa: E402


def synthetic(T=1200, D=6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)[:, None]
    x = 0.5 + 0.3 * np.sin(2 * np.pi * t / 40 + np.arange(D)) + 0.02 * rng.normal(size=(T, D))
    train, test = x[:800].astype(np.float32), x[800:].copy().astype(np.float32)
    labels = np.zeros(len(test), dtype=int)
    test[150:170] += 1.0
    labels[150:170] = 1
    return train, test, labels


CFG_FAST = dict(epochs=5, latent=4, hidden=32)


def test_m2n2_scores_anomaly_higher():
    train, test, labels = synthetic()
    det = M2N2Adapter({"dataset": "x", "hparams": {"default": {**CFG_FAST, "gamma": 0.9, "lr": 0.01}}})
    det.fit(train)
    s = det.score(test)
    assert s.shape == (len(test),) and np.isfinite(s).all()
    assert s[labels == 1].mean() > 2 * s[labels == 0].mean()
    assert det.train_scores().shape == (len(train),)


@pytest.mark.parametrize("norm", ["ema", "revin"])
def test_m2n2_runs_with_both_normalisations(norm):
    train, test, _ = synthetic()
    det = M2N2Adapter({"dataset": "x", "normalization": norm, "hparams": {"default": CFG_FAST}})
    det.fit(train)
    assert np.isfinite(det.score(test)).all()


def test_candi_adapts_and_scores_anomaly_higher():
    train, test, labels = synthetic()
    cfg = {**CFG_FAST, "sana_hidden": 16, "heads": 2, "min_buffer": 4, "ridge": 1e-2}
    det = CANDIAdapter({"dataset": "x", "hparams": {"default": cfg}})
    det.fit(train)
    s = det.score(test)
    assert s.shape == (len(test),) and np.isfinite(s).all()
    assert s[labels == 1].mean() > 2 * s[labels == 0].mean()
    assert det.stats["n_windows"] > 0


def test_candi_pretrained_never_updates():
    train, test, _ = synthetic()
    cfg = {**CFG_FAST, "sana_hidden": 16, "heads": 2, "adapt": False}
    det = CANDIAdapter({"dataset": "x", "hparams": {"default": cfg}})
    det.fit(train)
    det.score(test)
    assert det.stats.get("n_updates", 0) == 0
