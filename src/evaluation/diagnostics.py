"""
Drift diagnostics — measure how chained predictions diverge from direct predictions.

drift_curve computes cosine distance between z_hat_direct and z_hat_chain at each
horizon k, returning a (K,) tensor.  Call it during eval/logging passes (no grad).
"""

import torch
import torch.nn.functional as F


class DriftDiagnostics:
    """Thin wrapper around drift_curve for use as a stateful diagnostic object."""

    def __init__(self, predictor):
        self.predictor = predictor

    def compute(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:       (batch, latent_dim)
            actions: (batch, K) long
        Returns:
            drift: (K,) cosine distance per horizon
        """
        direct = self.predictor.forward_all_horizons(z, actions)
        chain = self.predictor.chain(z, actions)
        return drift_curve(direct.detach(), chain)


def save_latent_snapshot(z: torch.Tensor, path: str) -> None:
    """Save a latent tensor snapshot to disk."""
    torch.save(z.detach().cpu(), path)


def drift_curve(
    z_hat_direct: torch.Tensor,
    z_hat_chain: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine distance between direct and chained predictions, per horizon k.

    Args:
        z_hat_direct: (batch, K, latent_dim) — single-pass direct predictions
        z_hat_chain:  (batch, K, latent_dim) — autoregressive chained predictions
    Returns:
        drift: (K,) — mean cosine distance at each horizon k = 1 .. K
    """
    cos_sim = F.cosine_similarity(z_hat_direct, z_hat_chain, dim=-1)  # (batch, K)
    return (1.0 - cos_sim).mean(dim=0)                                 # (K,)
