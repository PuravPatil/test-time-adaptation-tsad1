"""Smoke and behaviour tests for the M2N2 detector, matched to the official code (needs torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from audit.backbones import build_backbone          # noqa: E402
from audit.backbones.mlp import pretrain             # noqa: E402
from audit.methods.m2n2_adapter import M2N2Adapter   # noqa: E402


def synthetic(T=1200, D=6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)[:, None]
    x = np.sin(2 * np.pi * t / 40 + np.arange(D)) + 0.05 * rng.normal(size=(T, D))
    train, test = x[:800].astype(np.float32), x[800:].copy().astype(np.float32)
    labels = np.zeros(len(test), dtype=int)
    test[150:170] += 3.0
    labels[150:170] = 1
    return train, test, labels


FAST = dict(epochs=5, latent=4)


def det(**hp):
    norm = hp.pop("normalization", "ema")
    seed = hp.pop("seed", 0)
    return M2N2Adapter({"dataset": "x", "seed": seed, "normalization": norm,
                        "hparams": {"default": {**FAST, **hp}}})


def test_official_mlp_shape_for_smd():
    m = build_backbone("default", 5, 38, latent=16)
    shapes = [tuple(p.shape) for p in m.parameters() if p.ndim == 2]
    assert shapes == [(95, 190), (47, 95), (16, 47), (47, 16), (95, 47), (190, 95)]
    assert isinstance(m.encoder[-1], torch.nn.ReLU)          # ReLU on the latent, as in models/MLP.py
    assert m(torch.zeros(2, 5, 38)).shape == (2, 5, 38)


def test_pretrain_keeps_the_best_epoch():
    torch.manual_seed(0)
    m = build_backbone("default", 5, 3, latent=2)
    w = np.random.default_rng(0).normal(size=(64, 5, 3)).astype(np.float32)
    hist = pretrain(m, w, epochs=6, lr=5e-2, batch_size=8)
    with torch.no_grad():
        x = torch.as_tensor(w)
        assert len(hist) == 6
        # the kept weights are those of the epoch with the lowest training loss (up to in-epoch drift)
        assert ((m(x) - x) ** 2).mean().item() <= max(hist) + 1e-6


def test_scores_are_per_timestep_and_anomaly_scores_higher():
    train, test, labels = synthetic()
    d = det(gamma=0.9, lr=0.01)
    d.fit(train)
    s = d.score(test)
    assert s.shape == (len(test),) and np.isfinite(s).all()
    assert s[labels == 1].mean() > 2 * s[labels == 0].mean()
    assert d.train_scores().shape == (len(train),)
    # per-timestep scores vary inside a window (not one value copied 5 times)
    assert len(np.unique(s[:5])) > 1


def test_single_tau_is_the_training_quantile():
    train, _, _ = synthetic()
    d = det(tau_quantile=0.9)
    d.fit(train)
    assert d.tau == pytest.approx(np.quantile(d.train_scores(), 0.9))


@pytest.mark.parametrize("norm", ["ema", "revin", "none"])
def test_runs_with_every_normalization(norm):
    train, test, _ = synthetic()
    d = det(normalization=norm)
    d.fit(train)
    assert np.isfinite(d.score(test)).all()


@pytest.mark.parametrize("mask", ["timestep", "window"])
def test_both_update_masks_adapt(mask):
    train, test, labels = synthetic()
    d = det(update_mask=mask)
    d.fit(train)
    s = d.score(test)
    assert s[labels == 1].mean() > 2 * s[labels == 0].mean()
    st = d.stats
    assert st["update_mask"] == mask and 0 < st["n_updates"] <= st["n_windows"]
    assert 0 < st["normal_step_rate"] <= 1 and st["tau"] == d.tau


def test_no_update_when_nothing_is_below_tau():
    train, test, _ = synthetic()
    d = det()
    d.fit(train)
    d.tau = -1.0                                     # every timestep is an alarm
    before = [p.detach().clone() for p in d.model.parameters()]
    d.score(test)
    assert d.stats["n_updates"] == 0
    assert all(bool((a == b).all()) for a, b in zip(before, d.model.parameters()))


def test_timestep_and_window_masks_agree_when_everything_is_normal():
    """With every timestep below tau, Eq. 4 averaged over L is the window MSE: same steps."""
    train, test, _ = synthetic()
    out = {}
    for mask in ("timestep", "window"):
        torch.manual_seed(0)
        d = det(update_mask=mask)
        d.fit(train)
        d.tau = float("inf")
        out[mask] = d.score(test)
    assert np.allclose(out["timestep"], out["window"], rtol=1e-5, atol=1e-7)


def test_static_baseline_never_updates():
    train, test, labels = synthetic()
    d = det(adapt=False)
    d.fit(train)
    before = [p.detach().clone() for p in d.model.parameters()]
    s = d.score(test)
    assert all(bool((a == b).all()) for a, b in zip(before, d.model.parameters()))
    assert d.stats["n_updates"] == 0 and s[labels == 1].mean() > 2 * s[labels == 0].mean()


def test_trend_starts_at_zero_and_follows_a_level_shift():
    """Official Detrender: mean starts at 0 (the standardised training mean) and is updated
    before each window is scored, so after a lasting shift the scores come back down."""
    train, test, _ = synthetic()
    test = test.copy()
    test[200:] += 2.0                                  # a new normal
    d = det(gamma=0.9, adapt=False)
    d.fit(train)
    s = d.score(test)
    assert s[200:210].mean() > 5 * s[350:].mean()      # spike at the shift, then tracked


def test_pretraining_shuffle_follows_the_run_seed():
    train, _, _ = synthetic()

    def scores(seed):
        torch.manual_seed(0)                           # same weight init: only the shuffle seed varies
        d = det(seed=seed, adapt=False)
        d.fit(train)
        return d.train_scores()

    assert np.array_equal(scores(1), scores(1))
    assert not np.array_equal(scores(1), scores(2))


def test_rejects_unknown_settings():
    with pytest.raises(ValueError):
        det(update_mask="segment")
    with pytest.raises(ValueError):
        det(normalization="zscore")
