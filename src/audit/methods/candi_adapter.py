"""CANDI (AAAI 2026, arXiv 2604.01845) behind the Detector interface.

Re-implemented from the paper's description; third_party/CANDI is never edited.
Equation/section names follow the paper.

  Pre-training   MLP autoencoder (L=10), Adam 1e-3, cosine, 30 epochs, on the
                 first 80% of the training data; the last 20% is the validation set.
  Threshold      tau = Percentile(S_val, 1 - alpha), alpha in {0.5%, 1%, 5%}.
  FPM            False Positive Mining. Latents of the FROZEN encoder are
                 L2-normalised. R_fp = validation latents with score > tau, R_mod =
                 validation latents with score in [Q1, Q3]. Mahalanobis distance uses
                 the covariance of all validation latents. A test window joins the
                 adaptation set A if
                   score > tau  and min_r D2(z, R_fp)  < delta   (hard set), or
                   score <= tau and min_r D2(z, R_mod) < delta   (moderate set),
                 with delta = chi2.ppf(0.05, d).
  SANA           Spatiotemporally-aware normality adaptation: a gated residual
                 block (temporal conv -> inter-variable attention -> linear) on the
                 detector's input and on its output. ONLY SANA is trained at test
                 time (SGD, Nesterov 0.9, wd 1e-4, grad-clip 0.5); the backbone
                 stays frozen so the latent space used by FPM does not drift.
  Streaming      windows arrive in batches of 256; adaptation fires once >= 16
                 selected windows have accumulated.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import chi2
from torch import nn

from audit.backbones import build_backbone
from audit.backbones.mlp import pretrain, score_windows, window_mse
from audit.data.windows import make_windows, spread_scores

from .base import Detector, resolve_hparams

DEFAULTS = dict(window=10, latent=16, hidden=128, epochs=30, pretrain_lr=1e-3, batch_size=256,
                val_frac=0.2, alpha=0.05, sana_hidden=512, heads=4, kernel=3, gate_init=0.1,
                adapt_lr=0.01, adapt_steps=1, min_buffer=16, chi_q=0.05, ridge=1e-3, adapt=True,
                eval_chunk=64, max_adapt_batch=128)


class _SANASide(nn.Module):
    """Conv over time (per variable) -> attention across variables -> linear -> gated residual."""

    def __init__(self, n_features: int, length: int, hidden: int, heads: int, kernel: int, gate_init: float):
        super().__init__()
        self.conv = nn.Conv1d(1, hidden, kernel, padding=kernel // 2)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.proj = nn.Linear(hidden, length)
        self.gate = nn.Parameter(torch.full((n_features,), float(gate_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, D, L)
        B, D, L = x.shape
        h = self.conv(x.reshape(B * D, 1, L)).mean(dim=-1).reshape(B, D, -1)   # (B, D, hidden)
        h, _ = self.attn(h, h, h, need_weights=False)
        a = self.proj(h)                                                        # (B, D, L)
        return x + torch.tanh(self.gate)[None, :, None] * a


class SANA(nn.Module):
    def __init__(self, backbone: nn.Module, n_features: int, length: int, hidden: int, heads: int,
                 kernel: int, gate_init: float):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.inp = _SANASide(n_features, length, hidden, heads, kernel, gate_init)
        self.out = _SANASide(n_features, length, hidden, heads, kernel, gate_init)

    def adapter_parameters(self):
        return list(self.inp.parameters()) + list(self.out.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, L, D)
        x_t = self.inp(x.transpose(1, 2)).transpose(1, 2)
        x_hat = self.backbone(x_t)
        return self.out(x_hat.transpose(1, 2)).transpose(1, 2)


class CANDIAdapter(Detector):
    name = "candi"

    def __init__(self, config: dict):
        super().__init__(config)
        self.hp = resolve_hparams(config, DEFAULTS)
        self.stats: dict = {}
        self._val_scores = None

    # -- helpers ---------------------------------------------------------
    @torch.no_grad()
    def _latent(self, windows: torch.Tensor) -> torch.Tensor:
        z = self.backbone.encode(windows)
        return nn.functional.normalize(z, dim=1)

    def _whiten(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mu) @ self.whitener

    # -- Detector API ----------------------------------------------------
    def fit(self, train: np.ndarray) -> None:
        hp = self.hp
        L, D = hp["window"], train.shape[1]
        n_fit = int(len(train) * (1 - hp["val_frac"]))
        fit_w, _ = make_windows(train[:n_fit], L)
        val_w, val_ends = make_windows(train[n_fit:], L)

        self.backbone = build_backbone(self.config.get("backbone", "default"), L, D,
                                       latent=hp["latent"], hidden=hp["hidden"])
        pretrain(self.backbone, fit_w, epochs=hp["epochs"], lr=hp["pretrain_lr"], batch_size=hp["batch_size"])
        self.backbone.eval()

        s_val = score_windows(self.backbone, val_w)
        self._val_scores = s_val
        self.tau = float(np.quantile(s_val, 1 - hp["alpha"]))

        # Reference sets in the frozen, L2-normalised latent space.
        z_val = self._latent(torch.as_tensor(val_w))
        self.mu = z_val.mean(dim=0)
        cov = torch.cov((z_val - self.mu).T.double())
        cov = cov + hp["ridge"] * cov.diagonal().mean() * torch.eye(cov.shape[0], dtype=cov.dtype)
        evals, evecs = torch.linalg.eigh(cov)
        self.whitener = (evecs @ torch.diag(evals.clamp_min(1e-12).rsqrt()) @ evecs.T).float()
        s = torch.as_tensor(s_val)
        q1, q3 = np.quantile(s_val, [0.25, 0.75])
        self.r_fp = self._whiten(z_val[s > self.tau])
        self.r_mod = self._whiten(z_val[(s >= float(q1)) & (s <= float(q3))])
        self.delta = float(chi2.ppf(hp["chi_q"], df=z_val.shape[1]))

        self.model = SANA(self.backbone, D, L, hp["sana_hidden"], hp["heads"], hp["kernel"], hp["gate_init"])
        self.stats = {"tau": self.tau, "delta": self.delta, "n_ref_fp": len(self.r_fp), "n_ref_mod": len(self.r_mod)}

    def train_scores(self) -> np.ndarray | None:
        """Validation-window scores from the pre-trained detector (no adaptation)."""
        return self._val_scores

    @staticmethod
    def _min_d2(z: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if len(ref) == 0:
            return torch.full((len(z),), float("inf"))
        return torch.cdist(z, ref).min(dim=1).values ** 2

    def score(self, test: np.ndarray) -> np.ndarray:
        hp = self.hp
        windows, ends = make_windows(test, hp["window"], 1)
        data = torch.as_tensor(windows)
        scores = np.empty(len(windows))
        opt = torch.optim.SGD(self.model.adapter_parameters(), lr=hp["adapt_lr"], momentum=0.9,
                              nesterov=True, weight_decay=1e-4)
        buffer: list[torch.Tensor] = []
        n_hard = n_mod = n_updates = 0
        bs = hp["batch_size"]
        self.model.eval()
        for i in range(0, len(data), bs):
            x = data[i:i + bs]
            with torch.no_grad():       # sub-batches keep the (B*D, hidden, L) conv activations small
                s = torch.cat([window_mse(c, self.model(c)) for c in x.split(hp["eval_chunk"])])
            scores[i:i + bs] = s.numpy()
            if not hp["adapt"]:
                continue
            z = self._whiten(self._latent(x))
            hard = (s > self.tau) & (self._min_d2(z, self.r_fp) < self.delta)
            mod = (s <= self.tau) & (self._min_d2(z, self.r_mod) < self.delta)
            sel = hard | mod
            n_hard += int(hard.sum())
            n_mod += int(mod.sum())
            if sel.any():
                buffer.append(x[sel])
            if sum(len(b) for b in buffer) >= hp["min_buffer"]:
                batch = torch.cat(buffer)
                buffer = []
                if len(batch) > hp["max_adapt_batch"]:       # bound memory/time per adaptation step
                    batch = batch[torch.randperm(len(batch))[:hp["max_adapt_batch"]]]
                self.model.train()
                for _ in range(hp["adapt_steps"]):
                    loss = window_mse(batch, self.model(batch)).mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.adapter_parameters(), 0.5)
                    opt.step()
                self.model.eval()
                n_updates += 1
        self.stats.update(n_hard=n_hard, n_moderate=n_mod, n_updates=n_updates, n_windows=len(data))
        return spread_scores(scores, ends, len(test))
