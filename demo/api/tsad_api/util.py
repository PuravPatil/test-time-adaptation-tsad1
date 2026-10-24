"""Small helpers: safe path parts, JSON-safe values, downsampling for charts."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")
_HEX_ID = re.compile(r"^[0-9a-f]{32}$")


def safe_name(value: str) -> bool:
    """A path part from a URL that may name a folder: letters, digits, _ . - ; no '..'."""
    return bool(_NAME.match(value)) and ".." not in value


def safe_id(value: str) -> bool:
    return bool(_HEX_ID.match(value))


def inside(root: Path, path: Path) -> bool:
    """True if path resolves to somewhere under root (guards against traversal)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def clean(obj):
    """Make a value JSON-safe: NaN/inf -> None, numpy scalars/arrays -> Python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def downsample(x, points: int) -> list[float]:
    """At most `points` values, each the MAX of its bucket, so short spikes and short
    anomaly segments stay visible in a chart."""
    x = np.asarray(x, dtype=float)
    if len(x) <= points:
        return x.tolist()
    edges = np.linspace(0, len(x), points + 1).astype(int)[:-1]
    return np.maximum.reduceat(x, edges).tolist()


def series_payload(scores, preds, labels, points: int) -> dict:
    n = len(scores)
    return clean({
        "n": n,
        "points": min(n, points),
        "bucket": max(1.0, n / points),
        "scores": downsample(scores, points),
        "preds": downsample(preds, points),
        "labels": downsample(labels, points) if labels is not None else None,
    })
