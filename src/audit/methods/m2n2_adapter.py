"""Wraps upstream M2N2 (third_party/M2N2) behind the Detector interface.

Rule: never edit third_party/. Import from it, and put any change to
upstream behaviour here so it is visible in review.
"""
from __future__ import annotations

import numpy as np

from .base import Detector


class M2N2Adapter(Detector):
    name = "m2n2"

    def fit(self, train: np.ndarray) -> None:
        # TODO (weeks 1-3): build the upstream model with self.config["hparams"]
        # and the normalization in self.config["normalization"] (ema | revin).
        raise NotImplementedError("M2N2 adapter not wired up yet")

    def score(self, test: np.ndarray) -> np.ndarray:
        # TODO: run upstream test-time adaptation in streaming order and
        # return one score per timestep.
        raise NotImplementedError("M2N2 adapter not wired up yet")
