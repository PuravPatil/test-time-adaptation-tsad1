"""No normalization: the model sees the raw (min-max scaled) window.

This is the "DT off" arm of the M2N2 ablation (paper Table 3). Same interface
as EMA and RevIN: ``step(window) -> (shift, scale)``, here always (0, 1).
"""
from __future__ import annotations

import numpy as np


class NoNormalization:
    def __init__(self, **_):
        pass

    def step(self, window: np.ndarray):
        return 0.0, 1.0
