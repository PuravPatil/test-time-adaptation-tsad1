"""Server Machine Dataset (Su et al., KDD 2019) loader.

Dataset names look like ``smd_1-7`` and map to machine-1-7. Data is read from
data/processed/smd/machine-1-7/{train,test,test_labels}.npy (see
scripts/preprocess.py). Values are standardised (zero mean, unit variance per
feature) with statistics fitted on the training split only, as in the official
M2N2 code (sklearn StandardScaler, fitted on train).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .registry import Dataset, register


def standardize_fit(train: np.ndarray):
    """Per-feature mean and std of the training data. A constant feature gets std 1
    (sklearn's StandardScaler does the same), so it maps to 0 instead of dividing by 0."""
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    return mu, np.where(sd == 0, 1.0, sd).astype(train.dtype)


def minmax_fit(train: np.ndarray, eps: float = 1e-8):
    """Min-max statistics (our earlier scaling; kept for reference)."""
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
    mu, sd = standardize_fit(train)
    return Dataset(f"smd_{entity.replace('machine-', '')}", (train - mu) / sd, (test - mu) / sd, labels)
