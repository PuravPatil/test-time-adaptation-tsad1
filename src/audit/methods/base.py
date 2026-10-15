"""The one interface every method sits behind.

The harness only ever calls fit() and score(), so it never needs to know
which paper it is running. Thresholding happens outside the detector,
which is what lets one streaming threshold work for every method.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    name: str = "detector"

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def fit(self, train: np.ndarray) -> None:
        """Train on (T_train, D) data assumed to be normal."""

    @abstractmethod
    def score(self, test: np.ndarray) -> np.ndarray:
        """Process the test stream in time order and return one anomaly
        score per timestep, shape (T_test,). Any test-time adaptation the
        method does happens inside this call, in streaming order."""

    def train_scores(self) -> np.ndarray | None:
        """Scores on the training data, computed during fit() with no
        test-time adaptation. Thresholds calibrate on these. Return None
        if the method cannot provide them."""
        return None


def resolve_hparams(config: dict, defaults: dict) -> dict:
    """defaults < config['hparams']['default'] < config['hparams'][<dataset>].

    The harness puts the dataset name into config['dataset']."""
    hp = dict(defaults)
    table = config.get("hparams") or {}
    hp.update(table.get("default", {}))
    hp.update(table.get(config.get("dataset", ""), {}))
    return hp
