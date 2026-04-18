"""
Auxiliary loss functions.

L_pred  — cosine similarity between direct prediction and stop-gradient target.
L_cons  — cosine similarity between chained and stop-gradient direct prediction.

CRITICAL stop-gradient directions (from implementation plan):
  L_pred: stop-gradient goes on the GROUND-TRUTH target z_{t+k}, not the prediction.
  L_cons: stop-gradient goes on the DIRECT prediction (the anchor), not the chained one.
Getting either backwards defeats the purpose of the loss.
"""

import torch
import torch.nn.functional as F

N_OBJECT_TYPES = 11  # MiniGrid object type range: 0-10


def prediction_loss(
    z_hat_direct: torch.Tensor,
    z_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-horizon direct prediction loss.

    L_pred = (1/K) * sum_{k=1}^{K} (1 - cos(z_hat_direct_{t+k}, sg(z_{t+k})))

    Args:
        z_hat_direct: (batch, K, latent_dim)  — direct predictions for k=1..K
        z_target:     (batch, K, latent_dim)  — ground-truth latents z_{t+1..t+K}
                                                 stop-gradient applied here.
    Returns:
        loss:           scalar
        per_horizon:    (K,) — per-horizon cosine distance for logging
    """
    # Stop-gradient on the target
    target = z_target.detach()

    # Cosine distance: 1 - cos_sim,  shape (batch, K)
    cos_sim = F.cosine_similarity(z_hat_direct, target, dim=-1)
    per_step = 1.0 - cos_sim                              # (batch, K)

    per_horizon = per_step.mean(dim=0)                    # (K,)
    loss = per_horizon.mean()
    return loss, per_horizon


def consistency_loss(
    z_hat_chain: torch.Tensor,
    z_hat_direct: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Temporal consistency regularizer.

    L_cons = (1/(K-1)) * sum_{k=2}^{K} (1 - cos(z_hat_chain_{t+k}, sg(z_hat_direct_{t+k})))

    The stop-gradient is on the DIRECT prediction (the stable anchor).
    Gradients flow through the chained rollout only.

    Args:
        z_hat_chain:  (batch, K, latent_dim)  — chained predictions k=1..K
        z_hat_direct: (batch, K, latent_dim)  — direct predictions (stop-grad target)
    Returns:
        loss:        scalar
        per_horizon: (K-1,) — per-horizon distances for k=2..K
    """
    if z_hat_chain.shape[1] < 2:
        zero = z_hat_chain.new_tensor(0.0)
        return zero, zero.unsqueeze(0)

    # k=2..K (index 1 onward)
    chain = z_hat_chain[:, 1:, :]       # (batch, K-1, latent_dim)
    direct = z_hat_direct[:, 1:, :].detach()  # stop-gradient on anchor

    cos_sim = F.cosine_similarity(chain, direct, dim=-1)  # (batch, K-1)
    per_step = 1.0 - cos_sim

    per_horizon = per_step.mean(dim=0)   # (K-1,)
    loss = per_horizon.mean()
    return loss, per_horizon


def reconstruction_loss(
    logits: torch.Tensor,
    obs: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy reconstruction loss over MiniGrid object types.

    The decoder is trained to predict which object occupies each cell from the
    latent z.  Gradients flow through the decoder and back into the encoder,
    pushing z to preserve spatial structure beyond what PPO alone requires.

    Args:
        logits: (batch, N_OBJECT_TYPES, H, W) — GridDecoder output
        obs:    (batch, 3, H, W) float32 in [0, 1] — normalized MiniGrid obs.
                Channel 0 encodes object_type / 255.
    Returns:
        loss: scalar cross-entropy averaged over all cells and batch
    """
    # Recover integer object type from normalized channel 0.
    # Values are multiples of 1/255, so rounding is exact.
    target = (obs[:, 0, :, :] * 255.0).round().long().clamp(0, N_OBJECT_TYPES - 1)
    return F.cross_entropy(logits, target)
