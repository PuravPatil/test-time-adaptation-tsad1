"""Server Machine Dataset (Su et al., KDD 2019) loader.

Dataset names look like ``smd_1-7`` and map to machine-1-7. Data is read from
data/processed/smd/machine-1-7/{train,test,test_labels}.npy (see
scripts/preprocess.py). Values are min-max scaled with statistics fitted on the
training split only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .registry import Dataset, register


def minmax_fit(train: np.ndarray, eps: float = 1e-8):
    lo, hi = train.min(axis=0), train.max(axis=0)
    return lo, np.maximum(hi - lo, eps)


@register("smd")
def load_smd(processed_dir: str, entity: str = "machine-1-1") -> Dataset:
    root = Path(processed_dir) / "smd" / entity
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Run: python scripts/download_data.py && python scripts/preprocess.py"
        )
    train = np.load(root / "train.npy").astype(np.float32)
    test = np.load(root / "test.npy").astype(np.float32)
    labels = np.load(root / "test_labels.npy").astype(int)
    lo, rng = minmax_fit(train)
    return Dataset(f"smd_{entity.replace('machine-', '')}", (train - lo) / rng, (test - lo) / rng, labels)
