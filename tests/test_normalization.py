import numpy as np

from audit.normalization import make_normalizer


def test_ema_follows_the_window_mean():
    n = make_normalizer("ema", gamma=0.5)
    w = np.ones((5, 2), dtype=np.float32)
    n.step(w)                                   # first window initialises the trend
    shift, scale = n.step(3 * w)                # 0.5 * 1 + 0.5 * 3
    assert np.allclose(shift, 2.0) and scale == 1.0


def test_ema_gamma_one_never_moves():
    n = make_normalizer("ema", gamma=1.0)
    w = np.ones((5, 2), dtype=np.float32)
    n.step(w)
    assert np.allclose(n.step(10 * w)[0], 1.0)


def test_revin_gives_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    w = rng.normal(5, 3, (50, 4)).astype(np.float32)
    shift, scale = make_normalizer("revin").step(w)
    z = (w - shift) / scale
    assert np.allclose(z.mean(0), 0, atol=1e-5) and np.allclose(z.std(0), 1, atol=1e-3)
