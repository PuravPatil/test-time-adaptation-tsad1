"""Normalization options (Tanny's extension): EMA detrending vs RevIN."""
from .ema import EMADetrend
from .revin import RevIN


def make_normalizer(name: str, gamma: float = 0.99, **kw):
    if name == "ema":
        return EMADetrend(gamma=gamma, **kw)
    if name == "revin":
        return RevIN(**kw)
    raise KeyError(f"Unknown normalization '{name}' (ema | revin)")


__all__ = ["EMADetrend", "RevIN", "make_normalizer"]
