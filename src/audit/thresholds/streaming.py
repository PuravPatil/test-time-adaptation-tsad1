"""Streaming threshold that adapts online as the test stream arrives.
Owner: Purav.

Both papers keep tau fixed after training; this fills that gap. It uses only
scores seen so far (no look-ahead): step() decides on the CURRENT state, then
updates the state.

    tau_t = mean_t + k * std_t

mean/var are exponentially weighted (rate `alpha`) over scores that were NOT
flagged, so anomalies do not drag the threshold up. They start from the scores of
normal training data (fit), so the first test steps already have a sensible tau.
Config: k (default 3.0), alpha (default 0.01), min_std (default 1e-8).
"""
from __future__ import annotations

import numpy as np

from .base import Threshold


class StreamingThreshold(Threshold):
    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.k = float(self.config.get("k", 3.0))
        self.alpha = float(self.config.get("alpha", 0.01))
        self.min_std = float(self.config.get("min_std", 1e-8))
        self.mean: float | None = None
        self.var = 0.0

    def fit(self, reference_scores: np.ndarray) -> None:
        ref = np.asarray(reference_scores, dtype=float)
        self.mean, self.var = float(ref.mean()), float(ref.var())

    @property
    def tau(self) -> float:
        if self.mean is None:
            raise RuntimeError("StreamingThreshold needs fit() on reference scores first")
        return self.mean + self.k * max(np.sqrt(self.var), self.min_std)

    def step(self, score: float) -> int:
        flag = int(score > self.tau)
        if not flag:                       # only normal-looking scores update the state
            d = score - self.mean
            self.mean += self.alpha * d
            self.var = (1 - self.alpha) * (self.var + self.alpha * d * d)
        return flag
