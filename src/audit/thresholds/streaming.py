"""Streaming threshold that adapts online as the test stream arrives.
Owner: Purav.

Both papers keep tau fixed after training; this fills that gap. It must only
use scores seen so far (no look-ahead), so step() is the real implementation.
"""
from __future__ import annotations

import numpy as np

from .base import Threshold


class StreamingThreshold(Threshold):
    def fit(self, reference_scores: np.ndarray) -> None:
        # TODO (weeks 4-6): initialise state from normal-data scores.
        pass

    def step(self, score: float) -> int:
        # TODO (weeks 4-6): update the threshold with this score and return 0/1.
        raise NotImplementedError("Streaming threshold not implemented yet")
