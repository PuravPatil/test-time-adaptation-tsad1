"""Swappable backbone models. Look one up by name: ``build_backbone("m2n2", ...)``.

    default, m2n2   the official M2N2 MLP (n -> n/2 -> n/4 -> latent and back)
    mlp             a one-hidden-layer MLP autoencoder (our earlier version)

Add a backbone by writing a module and registering it here (Uno's extension).
Torch is imported lazily so metric/threshold code works without it.
"""
from __future__ import annotations

BACKBONES = ("default", "m2n2", "mlp")


def build_backbone(name: str, window: int, n_features: int, **kw):
    if name in ("default", "m2n2"):
        from .m2n2 import M2N2MLP
        return M2N2MLP(window, n_features, latent=kw.get("latent", 16))
    if name == "mlp":
        from .mlp import MLPAutoencoder
        return MLPAutoencoder(window, n_features, latent=kw.get("latent", 16), hidden=kw.get("hidden", 128))
    raise KeyError(f"Unknown backbone '{name}'. Known: {list(BACKBONES)}")
