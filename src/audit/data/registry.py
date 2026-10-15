"""One common format for every dataset, and a registry of loaders.

Each loader reads from data/processed/<name>/ and returns a Dataset.
Add a loader by writing a function and decorating it with @register("<name>").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Dataset:
    name: str
    train: np.ndarray        # (T_train, D) values, assumed normal (unlabelled)
    test: np.ndarray         # (T_test, D) values
    test_labels: np.ndarray  # (T_test,) 0 = normal, 1 = anomaly

    def __post_init__(self) -> None:
        if len(self.test) != len(self.test_labels):
            raise ValueError(f"{self.name}: test values and labels differ in length")


_LOADERS: dict[str, Callable[..., Dataset]] = {}


def register(name: str):
    def wrap(fn: Callable[[str], Dataset]):
        _LOADERS[name] = fn
        return fn
    return wrap


def load_dataset(name: str, processed_dir: str = "data/processed") -> Dataset:
    """Load by name. ``smd`` loads a default entity; ``smd_1-7`` loads
    machine-1-7 (anything after the first underscore is passed as the entity)."""
    base, _, entity = name.partition("_")
    if base not in _LOADERS:
        raise KeyError(
            f"No loader registered for '{base}'. Write one in src/audit/data/ "
            f"and decorate it with @register('{base}'). Known: {sorted(_LOADERS)}"
        )
    if entity:
        return _LOADERS[base](processed_dir, entity=f"machine-{entity}")
    return _LOADERS[base](processed_dir)


