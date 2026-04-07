"""
MHAC PPO Trainer.

Subclasses SB3's PPO to inject auxiliary losses (L_pred, L_cons) into each
update step.  The auxiliary losses are added to the PPO loss before the
optimizer step, so the encoder is updated jointly by all objectives.

Usage
-----
    trainer = MHACTrainer(
        env=vec_envs,
        predictor=predictor,
        lambda_pred=0.1,
        lambda_cons=0.1,
        horizon=5,
        ...
    )
    trainer.learn(total_timesteps=5_000_000)
"""

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from src.models.predictor import MHACPredictor
from src.models.policy import MHACFeaturesExtractor
from src.training.losses import prediction_loss, consistency_loss


class MHACTrainer(PPO):
    """PPO + MHAC auxiliary losses."""

    def __init__(
        self,
        *args,
        predictor: MHACPredictor,
        lambda_pred: float = 0.1,
        lambda_cons: float = 0.1,
        horizon: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.predictor = predictor.to(self.device)
        self.lambda_pred = lambda_pred
        self.lambda_cons = lambda_cons
        self.horizon = horizon

        # Add predictor parameters to the optimizer
        self.policy.optimizer.add_param_group(
            {"params": self.predictor.parameters()}
        )

    # ------------------------------------------------------------------
    # Override the core update step to inject auxiliary losses
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        SB3 calls this after each rollout collection.  We call the parent
        implementation and then compute + log auxiliary loss statistics.

        Note: SB3's train() already handles gradient steps internally.
        We override _compute_losses() to inject aux terms into the total loss.
        """
        # Attach aux config so _compute_losses can read it
        self._aux_config = {
            "lambda_pred": self.lambda_pred,
            "lambda_cons": self.lambda_cons,
            "horizon": self.horizon,
        }
        super().train()

    def _compute_losses(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_log_prob: torch.Tensor,
        old_values: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Extends SB3's loss computation with MHAC auxiliary terms.

        Called once per mini-batch inside the PPO update epochs.
        """
        # Standard PPO loss from parent
        loss, info = super()._compute_losses(
            obs, actions, advantages, returns, old_log_prob, old_values
        )

        if self.lambda_pred == 0.0 and self.lambda_cons == 0.0:
            return loss, info

        batch_size = obs.shape[0]
        K = self.horizon

        # We need sequential observations to build (z_t, a_t..a_{t+K-1}, z_{t+K})
        # tuples.  SB3 shuffles the rollout buffer for mini-batches, so we can
        # only use valid (t, t+K) pairs that fall within the same rollout.
        # This is handled in the training script via a custom rollout sampler.
        # When that data isn't available, skip the aux loss gracefully.
        aux_data = getattr(self, "_aux_batch", None)
        if aux_data is None:
            return loss, info

        z_t = aux_data["z_t"]                    # (B, latent_dim)
        action_seq = aux_data["action_seq"]       # (B, K)  long
        z_targets = aux_data["z_targets"]         # (B, K, latent_dim)

        # --- Direct predictions for all horizons ---
        z_hat_direct = self.predictor.forward_all_horizons(z_t, action_seq)
        # (B, K, latent_dim)

        aux_loss = loss.new_tensor(0.0)
        aux_info = {}

        if self.lambda_pred > 0.0:
            l_pred, per_h_pred = prediction_loss(z_hat_direct, z_targets)
            aux_loss = aux_loss + self.lambda_pred * l_pred
            aux_info["loss/aux_pred"] = l_pred.item()
            for k, v in enumerate(per_h_pred.tolist()):
                aux_info[f"loss/pred_k{k+1}"] = v

        if self.lambda_cons > 0.0:
            z_hat_chain = self.predictor.chain_with_grad(z_t, action_seq)
            # (B, K, latent_dim)
            l_cons, per_h_cons = consistency_loss(z_hat_chain, z_hat_direct)
            aux_loss = aux_loss + self.lambda_cons * l_cons
            aux_info["loss/aux_cons"] = l_cons.item()
            for k, v in enumerate(per_h_cons.tolist()):
                aux_info[f"loss/cons_k{k+2}"] = v

        info.update(aux_info)
        return loss + aux_loss, info
