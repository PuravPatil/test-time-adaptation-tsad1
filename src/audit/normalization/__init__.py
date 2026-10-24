"""Normalization options: EMA detrending (M2N2), RevIN (Tanny's extension), or none
(the detrend-off arm of the ablation)."""
from .ema import EMADetrend
from .identity import NoNormalization
from .revin import RevIN


def make_normalizer(name: str, gamma: float = 0.99, **kw):
    if name == "ema":
        return EMADetrend(gamma=gamma, **kw)
    if name == "revin":
        return RevIN(**kw)
    if name == "none":
        return NoNormalization(**kw)
    raise KeyError(f"Unknown normalization '{name}' (ema | revin | none)")


__all__ = ["EMADetrend", "NoNormalization", "RevIN", "make_normalizer"]
