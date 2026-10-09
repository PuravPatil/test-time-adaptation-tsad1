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


_LOADERS: dict[str, Callable[[str], Dataset]] = {}


def register(name: str):
    def wrap(fn: Callable[[str], Dataset]):
        _LOADERS[name] = fn
        return fn
    return wrap


def load_dataset(name: str, processed_dir: str = "data/processed") -> Dataset:
    if name not in _LOADERS:
        raise KeyError(
            f"No loader registered for '{name}'. Write one in src/audit/data/ "
            f"and decorate it with @register('{name}'). Known: {sorted(_LOADERS)}"
        )
    return _LOADERS[name](processed_dir)


# TODO (weeks 1-3): add loaders for smd, msl, smap, creditcard, e.g.
#
# @register("smd")
# def load_smd(processed_dir: str) -> Dataset:
#     ...
