"""MLP autoencoder used by both M2N2 and CANDI (CPU, PyTorch).

A window (L, D) is flattened to L*D, encoded to a d-dimensional latent and
decoded back. ``encode`` is exposed because CANDI measures distances in the
frozen latent space.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MLPAutoencoder(nn.Module):
    def __init__(self, window: int, n_features: int, latent: int = 16, hidden: int = 128):
        super().__init__()
        self.window, self.n_features = window, n_features
        flat = window * n_features
        self.encoder = nn.Sequential(nn.Linear(flat, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, flat))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, latent)"""
        return self.encoder(x.reshape(x.shape[0], -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> reconstruction (B, L, D)"""
        return self.decoder(self.encode(x)).reshape(x.shape)


def window_mse(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Per-window anomaly score: mean squared error over L*D values. -> (B,)"""
    return ((x - x_hat) ** 2).flatten(1).mean(dim=1)


def pretrain(model: nn.Module, windows: np.ndarray, epochs: int = 30, lr: float = 1e-3,
             batch_size: int = 256, seed: int = 0) -> list[float]:
    """Adam + cosine schedule, as in CANDI's pre-training. Returns per-epoch loss."""
    g = torch.Generator().manual_seed(seed)
    data = torch.as_tensor(windows, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    history = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(data), generator=g)
        total = 0.0
        for i in range(0, len(data), batch_size):
            batch = data[perm[i:i + batch_size]]
            loss = window_mse(batch, model(batch)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        sched.step()
        history.append(total / len(data))
    model.eval()
    return history


@torch.no_grad()
def score_windows(model: nn.Module, windows: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(windows), batch_size):
        x = torch.as_tensor(windows[i:i + batch_size], dtype=torch.float32)
        out.append(window_mse(x, model(x)).numpy())
    return np.concatenate(out)
