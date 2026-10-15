"""M2N2 (Kim, Park, Choo; AAAI 2024) behind the Detector interface.

Re-implemented from the paper (Algorithm 1 and appendix Table 6), because the
upstream repo is a GPU research script rather than an importable library.
third_party/M2N2 is never edited.

Pipeline
  1. Normalise each window with EMA detrending (or RevIN, Tanny's extension).
  2. MLP autoencoder trained on the (normalised) training stream.
  3. At test time, for each incoming window: score it, and if it is predicted
     normal (score < tau) take ONE plain-SGD step (batch size 1, no momentum)
     on its reconstruction loss. Windows predicted anomalous never update the
     model; the trend still follows the entire stream.
"""
from __future__ import annotations

import numpy as np
import torch

from audit.backbones import build_backbone
from audit.backbones.mlp import pretrain, window_mse
from audit.data.windows import make_windows, spread_scores
from audit.normalization import make_normalizer

from .base import Detector, resolve_hparams

DEFAULTS = dict(window=5, stride=5, latent=16, hidden=128, gamma=0.99, lr=0.05,
                epochs=30, batch_size=128, pretrain_lr=1e-3, tau_quantile=0.99, adapt=True)


class M2N2Adapter(Detector):
    name = "m2n2"

    def __init__(self, config: dict):
        super().__init__(config)
        self.hp = resolve_hparams(config, DEFAULTS)
        self.normalization = config.get("normalization", "ema")
        self.model = None
        self._train_scores = None
        self.tau = None

    # -- helpers ---------------------------------------------------------
    def _normalise_stream(self, x: np.ndarray, stride: int):
        """Windows normalised in stream order, advancing the normaliser state."""
        windows, ends = make_windows(x, self.hp["window"], stride)
        out = np.empty_like(windows)
        for i, w in enumerate(windows):
            shift, scale = self.norm.step(w)
            out[i] = (w - shift) / scale
        return out, ends

    def _window_scores(self, windows: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(windows)
            return window_mse(x, self.model(x)).numpy()

    # -- Detector API ----------------------------------------------------
    def fit(self, train: np.ndarray) -> None:
        hp = self.hp
        self.norm = make_normalizer(self.normalization, gamma=hp["gamma"])
        windows, ends = self._normalise_stream(train, hp["stride"])
        self.model = build_backbone(self.config.get("backbone", "default"), hp["window"],
                                    train.shape[1], latent=hp["latent"], hidden=hp["hidden"])
        pretrain(self.model, windows, epochs=hp["epochs"], lr=hp["pretrain_lr"],
                 batch_size=hp["batch_size"])
        ws = self._window_scores(windows)
        self._train_scores = spread_scores(ws, ends, len(train))
        self.tau = float(np.quantile(ws, hp["tau_quantile"]))

    def train_scores(self) -> np.ndarray | None:
        return self._train_scores

    def score(self, test: np.ndarray) -> np.ndarray:
        hp = self.hp
        windows, ends = make_windows(test, hp["window"], hp["stride"])
        opt = torch.optim.SGD(self.model.parameters(), lr=hp["lr"], momentum=0.0)
        scores = np.empty(len(windows))
        self.model.eval()
        for i, w in enumerate(windows):
            shift, scale = self.norm.step(w)
            x = torch.as_tensor(((w - shift) / scale)[None], dtype=torch.float32)
            recon = self.model(x)
            loss = window_mse(x, recon)[0]
            scores[i] = loss.item()
            if hp["adapt"] and scores[i] < self.tau:   # predicted normal -> update
                opt.zero_grad()
                loss.backward()
                opt.step()
        return spread_scores(scores, ends, len(test))
