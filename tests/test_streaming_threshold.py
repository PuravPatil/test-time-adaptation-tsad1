import numpy as np
import pytest

from audit.thresholds import StreamingThreshold


def test_flags_spike_not_normal_scores():
    rng = np.random.default_rng(0)
    thr = StreamingThreshold({"k": 3.0})
    thr.fit(rng.normal(1.0, 0.1, 500))
    assert thr.step(1.0) == 0
    assert thr.step(5.0) == 1


def test_no_lookahead():
    rng = np.random.default_rng(1)
    ref, stream = rng.normal(1, 0.1, 300), rng.normal(1, 0.1, 100)
    a, b = StreamingThreshold(), StreamingThreshold()
    a.fit(ref); b.fit(ref)
    full = a.predict(stream)
    # decisions on the first 50 steps must not depend on what comes later
    assert b.predict(stream[:50]).tolist() == full[:50].tolist()


def test_anomalies_do_not_raise_threshold():
    thr = StreamingThreshold()
    thr.fit(np.random.default_rng(2).normal(1, 0.1, 300))
    before = thr.tau
    for _ in range(50):
        thr.step(10.0)
    assert thr.tau == pytest.approx(before)


def test_requires_fit():
    with pytest.raises(RuntimeError):
        StreamingThreshold().step(1.0)
