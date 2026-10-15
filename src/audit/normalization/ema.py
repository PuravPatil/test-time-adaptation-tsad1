"""EMA detrending, as used by M2N2 (Kim et al., AAAI 2024).

Trend update per window (w = window length):
    mu_t <- gamma * mu_{t-w} + (1 - gamma) * mean(window)
and the model sees (window - mu_t). The trend uses the whole stream, including
windows later judged anomalous. Interface shared with RevIN:
``step(window) -> (shift, scale)`` and ``x_norm = (x - shift) / scale``.
"""
from __future__ import annotations

import numpy as np


class EMADetrend:
    def __init__(self, gamma: float = 0.99, init: np.ndarray | None = None):
        self.gamma = gamma
        self.mu = None if init is None else np.asarray(init, dtype=np.float32).copy()

    def step(self, window: np.ndarray):
        mean = window.mean(axis=0)
        self.mu = mean.copy() if self.mu is None else self.gamma * self.mu + (1 - self.gamma) * mean
        return self.mu, 1.0
