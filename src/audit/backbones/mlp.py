"""A one-hidden-layer MLP autoencoder, plus the shared training and scoring helpers (CPU, PyTorch).

The paper's own architecture is in m2n2.py; ``pretrain`` and ``window_mse`` work for any backbone.
"""
from __future__ import annotations

import copy

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


def pretrain(model: nn.Module, windows: np.ndarray, epochs: int = 10, lr: float = 1e-3,
             batch_size: int = 8, seed: int = 0, optimizer: str = "adamw",
             schedule: str = "none", keep_best: bool = True) -> list[float]:
    """Train on normal windows with MSE reconstruction loss. Returns per-epoch mean loss.

    Defaults follow the official M2N2 trainer (Exp/MLP.py): AdamW (weight decay 0),
    constant learning rate, shuffled mini-batches, and the weights of the epoch with
    the lowest training loss are kept ("best.pth"). ``optimizer="adam"`` and
    ``schedule="cosine"`` give our earlier setup.
    """
    g = torch.Generator().manual_seed(seed)
    data = torch.as_tensor(windows, dtype=torch.float32)
    if optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    elif optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"optimizer must be 'adamw' or 'adam', got {optimizer!r}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1)) if schedule == "cosine" else None
    history, best_loss, best_state = [], float("inf"), None
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(data), generator=g)
        total = 0.0
        for i in range(0, len(data), batch_size):
            batch = data[perm[i:i + batch_size]]
            loss = ((model(batch) - batch) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        if sched:
            sched.step()
        epoch_loss = total / len(data)
        history.append(epoch_loss)
        if keep_best and epoch_loss < best_loss:
            best_loss, best_state = epoch_loss, copy.deepcopy(model.state_dict())
    if keep_best and best_state is not None:
        model.load_state_dict(best_state)
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
