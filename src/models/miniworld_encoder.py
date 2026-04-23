"""
Nature-DQN style CNN encoder for MiniWorld RGB observations.

Input shape: (batch, 3, 60, 80) float32 in [0, 1]
Output shape: (batch, latent_dim)

Sized for (60, 80) inputs:
    Conv(3->32, k=8, s=4)  -> (32, 14, 19)
    Conv(32->64, k=4, s=2) -> (64,  6,  8)
    Conv(64->64, k=3, s=1) -> (64,  4,  6)
    Flatten -> 1536 -> Linear(latent_dim)
"""

import torch
import torch.nn as nn


class MiniWorldCNNEncoder(nn.Module):
    def __init__(self, obs_shape: tuple = (3, 60, 80), latent_dim: int = 256):
        super().__init__()
        c, h, w = obs_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_dim = self.net(dummy).shape[1]

        self.proj = nn.Linear(flat_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(obs))
