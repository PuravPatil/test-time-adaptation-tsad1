"""M2N2 (Kim, Park, Choo; AAAI 2024) behind the Detector interface.

Re-implemented to match the paper (Eq. 2-4) and, where the paper is silent, the
authors' official code (github.com/carrtesy/M2N2: models/MLP.py, models/Normalizer.py,
Exp/MLP.py, Exp/Tester.py). third_party/M2N2 is never edited.

Training
  Windows of the standardised training stream (length `window`, stride `stride`,
  the last window aligned to the end), the official MLP (backbones/m2n2.py),
  AdamW lr `pretrain_lr` for `epochs` epochs in batches of `batch_size`, keeping
  the epoch with the lowest training loss.
  The EMA trend is NOT updated while training: it stays at 0, which is the
  training mean after standardisation (the official Detrender starts at zeros
  and is only updated during online inference).

Threshold
  ONE tau: the `tau_quantile` of the per-timestep training scores. It decides the
  alarms (score >= tau) AND which timesteps the model learns from (score < tau).
  The harness reads it from ``stats["tau"]`` when an experiment sets
  ``threshold_config: {quantile: detector}``.

Test time, window by window in stream order
  1. trend  mu <- gamma * mu + (1 - gamma) * mean(window)      (Eq. 2; normalization "ema")
  2. reconstruct the detrended window and add the trend back
  3. per-timestep score  A_t = mean over features of (x_hat_t - x_t)^2
  4. one plain-SGD step (lr `lr`) on  mean_t[ 1(A_t < tau) * A_t ]   (Eq. 3-4)
  The trend follows every window; only timesteps predicted normal update the model.

normalization "none" gives the paper's MLP and TTA-only rows; "revin" is Tanny's
extension (per-window mean/std, also applied in training). update_mask "window"
is our earlier variant (update on the whole window only if its mean score < tau).
"""
from __future__ import annotations

import numpy as np
import torch

from audit.backbones import build_backbone
from audit.backbones.mlp import pretrain
from audit.data.windows import make_windows, spread_step_scores
from audit.normalization import make_normalizer

from .base import Detector, resolve_hparams

DEFAULTS = dict(window=5, stride=5, latent=16, gamma=0.99, lr=0.05, epochs=10, batch_size=8,
                pretrain_lr=1e-3, tau_quantile=0.99, adapt=True, update_mask="timestep")
UPDATE_MASKS = ("timestep", "window")
NORMALIZATIONS = ("ema", "none", "revin")


class M2N2Adapter(Detector):
    name = "m2n2"

    def __init__(self, config: dict):
        super().__init__(config)
        self.hp = resolve_hparams(config, DEFAULTS)
        self.normalization = config.get("normalization", "ema")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {NORMALIZATIONS}, got {self.normalization!r}")
        if self.hp["update_mask"] not in UPDATE_MASKS:
            raise ValueError(f"update_mask must be one of {UPDATE_MASKS}, got {self.hp['update_mask']!r}")
        self.model = None
        self._train_scores = None
        self.tau = None
        self.stats: dict = {}   # filled by score(); the harness saves it in run.json

    # -- helpers ---------------------------------------------------------
    def _train_view(self, windows: np.ndarray) -> np.ndarray:
        """What the network sees in training. EMA: trend = 0, so the raw windows; RevIN: per window."""
        if self.normalization != "revin":
            return windows
        norm = make_normalizer("revin")
        out = np.empty_like(windows)
        for i, w in enumerate(windows):
            shift, scale = norm.step(w)
            out[i] = (w - shift) / scale
        return out

    def _step_errors(self, x: np.ndarray) -> np.ndarray:
        """Per-timestep reconstruction error (mean over features). (N, L, D) -> (N, L)"""
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float32)
            return ((self.model(t) - t) ** 2).mean(dim=2).numpy()

    # -- Detector API ----------------------------------------------------
    def fit(self, train: np.ndarray) -> None:
        hp = self.hp
        windows, ends = make_windows(train, hp["window"], hp["stride"])
        x = self._train_view(windows)
        self.model = build_backbone(self.config.get("backbone", "default"), hp["window"],
                                    train.shape[1], latent=hp["latent"], hidden=hp.get("hidden", 128))
        # The run seed (set by the harness) drives the mini-batch shuffle; weight
        # init is seeded globally by audit.utils.seed.set_seed before this call.
        pretrain(self.model, x, epochs=hp["epochs"], lr=hp["pretrain_lr"], batch_size=hp["batch_size"],
                 seed=int(self.config.get("seed", 0)), optimizer=hp.get("optimizer", "adamw"),
                 schedule=hp.get("schedule", "none"), keep_best=hp.get("keep_best", True))
        self._train_scores = spread_step_scores(self._step_errors(x), ends, len(train))
        self.tau = float(np.quantile(self._train_scores, hp["tau_quantile"]))

    def train_scores(self) -> np.ndarray | None:
        return self._train_scores

    def score(self, test: np.ndarray) -> np.ndarray:
        hp = self.hp
        windows, ends = make_windows(test, hp["window"], hp["stride"])
        # a fresh trend at 0 = the training mean (see the module docstring)
        norm = make_normalizer(self.normalization, gamma=hp["gamma"], init=np.zeros(test.shape[1], np.float32))
        opt = torch.optim.SGD(self.model.parameters(), lr=hp["lr"], momentum=0.0)
        step_scores = np.empty((len(windows), hp["window"]))
        n_updates, n_steps_used = 0, 0
        self.model.eval()
        for i, w in enumerate(windows):
            shift, scale = norm.step(w)                       # trend update happens BEFORE scoring
            x = torch.as_tensor(((w - shift) / scale)[None], dtype=torch.float32)
            err = ((self.model(x) - x) ** 2).mean(dim=2)[0]   # A_t, shape (L,)
            step_scores[i] = err.detach().numpy()
            if not hp["adapt"]:
                continue
            if hp["update_mask"] == "window":
                used = len(err) if err.mean().item() < self.tau else 0
                update = err.mean() if used else None
            else:                                             # Eq. 4: (1 - y_hat)^T A, averaged over L
                normal = (err.detach() < self.tau).float()
                used = int(normal.sum().item())
                update = (normal * err).mean() if used else None
            if update is not None:
                opt.zero_grad()
                update.backward()
                opt.step()
                n_updates += 1
                n_steps_used += used
        n_steps = len(windows) * hp["window"]
        self.stats = {"n_windows": len(windows), "n_updates": n_updates,
                      "update_rate": n_updates / max(len(windows), 1),
                      "normal_step_rate": n_steps_used / max(n_steps, 1),
                      "update_mask": hp["update_mask"], "tau": self.tau,
                      "tau_quantile": hp["tau_quantile"]}
        return spread_step_scores(step_scores, ends, len(test))
