"""Thresholds turn anomaly scores into 0/1 predictions, outside the method."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Threshold(ABC):
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def fit(self, reference_scores: np.ndarray) -> None:
        """Optional: calibrate on scores from normal (training) data."""

    @abstractmethod
    def step(self, score: float) -> int:
        """Consume one score in stream order and return 0 or 1."""

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.array([self.step(float(s)) for s in scores], dtype=int)
