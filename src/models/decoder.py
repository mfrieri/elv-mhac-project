"""
GridDecoder: latent_dim -> (N_OBJECT_TYPES, H, W) logits.

Decodes a 256-dim latent back to a per-cell object-type distribution over the
7x7 MiniGrid grid.  Used to (a) add a reconstruction auxiliary loss that trains
the encoder to preserve spatial structure, and (b) visualize what the agent
"imagines" the world looks like at future time steps via decoded predicted latents.

MiniGrid object type index (channel 0 of the raw obs, values 0-10):
  0=unseen  1=empty   2=wall   3=floor  4=door
  5=key     6=ball    7=box    8=goal   9=lava  10=agent
"""

import torch
import torch.nn as nn

N_OBJECT_TYPES = 11  # values 0-10 in MiniGrid


class GridDecoder(nn.Module):
    def __init__(self, latent_dim: int = 256, grid_h: int = 7, grid_w: int = 7):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, grid_h * grid_w * N_OBJECT_TYPES),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            logits: (batch, N_OBJECT_TYPES, grid_h, grid_w)  — compatible with F.cross_entropy
        """
        B = z.shape[0]
        return (
            self.net(z)
            .view(B, self.grid_h, self.grid_w, N_OBJECT_TYPES)
            .permute(0, 3, 1, 2)
        )

    def decode_to_grid(self, z: torch.Tensor) -> torch.Tensor:
        """Return argmax object-type predictions as (batch, H, W) int64."""
        with torch.no_grad():
            return self.forward(z).argmax(dim=1)
