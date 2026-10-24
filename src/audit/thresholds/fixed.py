"""Fixed threshold tau, chosen once and never updated (as in the paper).

tau can be given directly, set as a quantile of reference (training) scores, or
taken from the detector (``quantile: detector`` in an experiment's
threshold_config; the harness passes the detector's own tau, which for M2N2 is
the same tau that decides which timesteps it adapts on).

A score AT OR ABOVE tau is an alarm, as in the official M2N2 code (A >= tau).
"""
from __future__ import annotations

import numpy as np

from .base import Threshold


class FixedThreshold(Threshold):
    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.tau = self.config.get("tau")
        self.quantile = self.config.get("quantile", 0.99)

    def fit(self, reference_scores: np.ndarray) -> None:
        if self.tau is None:
            self.tau = float(np.quantile(reference_scores, self.quantile))

    def step(self, score: float) -> int:
        if self.tau is None:
            raise RuntimeError("FixedThreshold needs tau or a call to fit() first")
        return int(score >= self.tau)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.tau is None:
            raise RuntimeError("FixedThreshold needs tau or a call to fit() first")
        return (np.asarray(scores) >= self.tau).astype(int)
