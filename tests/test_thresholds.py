import numpy as np

from audit.thresholds import FixedThreshold


def test_fixed_threshold_with_tau():
    thr = FixedThreshold({"tau": 0.5})
    assert thr.predict(np.array([0.1, 0.5, 0.9])).tolist() == [0, 0, 1]


def test_fixed_threshold_from_quantile():
    thr = FixedThreshold({"quantile": 0.5})
    thr.fit(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert thr.tau == 2.0
    assert thr.step(2.5) == 1
