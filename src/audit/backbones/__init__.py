"""Swappable backbone models. Look one up by name: ``build_backbone("mlp", ...)``.

Add a backbone by writing a module and registering it in BACKBONES (Uno's
extension). Torch is imported lazily so metric/threshold code works without it.
"""
from __future__ import annotations


def build_backbone(name: str, window: int, n_features: int, **kw):
    if name in ("default", "mlp"):
        from .mlp import MLPAutoencoder
        return MLPAutoencoder(window, n_features, **kw)
    raise KeyError(f"Unknown backbone '{name}'. Known: ['mlp']")
