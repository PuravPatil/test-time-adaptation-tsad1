"""Reversible Instance Normalization (RevIN, Kim et al. 2022) without the
learnable affine part, as a drop-in for EMA detrending. Same interface:
``step(window) -> (shift, scale)``, ``x_norm = (x - shift) / scale``.
Each window is normalised by its own mean and std, so there is no state."""
from __future__ import annotations

import numpy as np


class RevIN:
    def __init__(self, eps: float = 1e-5, **_):
        self.eps = eps

    def step(self, window: np.ndarray):
        return window.mean(axis=0), np.sqrt(window.var(axis=0) + self.eps)
