"""
CNN Encoder: (C, H, W) -> latent_dim

Two convolutional layers followed by a linear projection to a 256-dim latent
vector.  Intentionally small — MiniGrid observations are simple (7x7x3) and a
large encoder will overfit, making the auxiliary prediction task redundant.

Input shape: (batch, 3, 7, 7)  float32 in [0, 1]
Output shape: (batch, latent_dim)
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, obs_shape: tuple = (3, 7, 7), latent_dim: int = 256):
        super().__init__()
        c, h, w = obs_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_dim = self.net(dummy).shape[1]

        self.proj = nn.Linear(flat_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, C, H, W) float32
        Returns:
            z: (batch, latent_dim)
        """
        return self.proj(self.net(obs))
