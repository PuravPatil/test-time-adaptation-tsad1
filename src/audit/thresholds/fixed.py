"""Fixed threshold tau, chosen once and never updated (as in both papers).

TODO (weeks 1-3): match how each paper picks tau so the reproduction is fair.
For now, tau can be given directly or set as a quantile of reference scores.
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
        return int(score > self.tau)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.tau is None:
            raise RuntimeError("FixedThreshold needs tau or a call to fit() first")
        return (np.asarray(scores) > self.tau).astype(int)
