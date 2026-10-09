"""Wraps upstream CANDI (third_party/CANDI) behind the Detector interface.

Rule: never edit third_party/. Import from it, and put any change to
upstream behaviour here so it is visible in review.
"""
from __future__ import annotations

import numpy as np

from .base import Detector


class CANDIAdapter(Detector):
    name = "candi"

    def fit(self, train: np.ndarray) -> None:
        # TODO (weeks 1-3): build the upstream model with self.config["hparams"].
        raise NotImplementedError("CANDI adapter not wired up yet")

    def score(self, test: np.ndarray) -> np.ndarray:
        # TODO: run upstream test-time adaptation in streaming order and
        # return one score per timestep.
        raise NotImplementedError("CANDI adapter not wired up yet")
