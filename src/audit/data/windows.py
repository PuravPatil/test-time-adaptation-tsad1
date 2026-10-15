"""Sliding windows and the window-score -> per-timestep-score mapping."""
from __future__ import annotations

import numpy as np


def window_ends(T: int, length: int, stride: int) -> np.ndarray:
    """Index of the last timestep of every window. The final window is always
    aligned to the end of the series so every timestep is covered."""
    if T < length:
        raise ValueError(f"series of length {T} is shorter than window {length}")
    ends = list(range(length - 1, T, stride))
    if ends[-1] != T - 1:
        ends.append(T - 1)
    return np.asarray(ends, dtype=int)


def make_windows(x: np.ndarray, length: int, stride: int = 1):
    """x: (T, D) -> windows (N, length, D) and their end indices (N,)."""
    ends = window_ends(len(x), length, stride)
    idx = ends[:, None] + np.arange(-length + 1, 1)[None, :]
    return x[idx].astype(np.float32), ends


def spread_scores(window_scores: np.ndarray, ends: np.ndarray, T: int) -> np.ndarray:
    """One score per timestep. A window's score covers every timestep after the
    previous window's end up to its own end (the first window covers 0..end)."""
    out = np.zeros(T, dtype=float)
    prev = -1
    for s, e in zip(window_scores, ends):
        out[prev + 1:e + 1] = s
        prev = e
    return out
