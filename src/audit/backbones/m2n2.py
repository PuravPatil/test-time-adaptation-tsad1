"""The MLP autoencoder exactly as in the official M2N2 code (models/MLP.py).

A window (L, D) is flattened to n = L*D and passed through

    encoder  n -> n//2 -> n//4 -> latent   (ReLU after every layer, the latent included)
    decoder  latent -> n//4 -> n//2 -> n   (ReLU after the first two layers)

For SMD (L = 5, D = 38): 190 -> 95 -> 47 -> 16 -> 47 -> 95 -> 190.
"""
from __future__ import annotations

import torch
from torch import nn


class M2N2MLP(nn.Module):
    def __init__(self, window: int, n_features: int, latent: int = 16, **_):
        super().__init__()
        self.window, self.n_features = window, n_features
        n = window * n_features
        self.encoder = nn.Sequential(
            nn.Linear(n, n // 2), nn.ReLU(),
            nn.Linear(n // 2, n // 4), nn.ReLU(),
            nn.Linear(n // 4, latent), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, n // 4), nn.ReLU(),
            nn.Linear(n // 4, n // 2), nn.ReLU(),
            nn.Linear(n // 2, n),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, latent)"""
        return self.encoder(x.reshape(x.shape[0], -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> reconstruction (B, L, D)"""
        return self.decoder(self.encode(x)).reshape(x.shape)
