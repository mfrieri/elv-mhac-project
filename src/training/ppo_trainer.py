"""
MHAC PPO Trainer.

For Phase 2 (baseline), this is just SB3's PPO with the custom CNN encoder
plugged in via policy_kwargs.  Aux loss injection (Phases 3–6) is added in
train() by running a second backward pass over sequential rollout tuples after
the standard PPO update.

Correct SB3 integration points:
  - Custom encoder → policy_kwargs["features_extractor_class"]
  - Predictor params → added in _setup_model() after policy is built
  - Aux losses      → second optimizer step in train() over sequential tuples
"""

import torch
import numpy as np
from stable_baselines3 import PPO

from src.models.policy import MHACFeaturesExtractor
from src.training.losses import prediction_loss, consistency_loss


class MHACTrainer(PPO):
    """
    PPO with optional MHAC auxiliary losses.

    Pass predictor=None and lambda_pred=0 for a pure baseline run.
    """

    def __init__(
        self,
        *args,
        predictor=None,
        lambda_pred: float = 0.0,
        lambda_cons: float = 0.0,
        horizon: int = 5,
        **kwargs,
    ):
        # Inject our feature extractor so the CNN encoder is used automatically
        policy_kwargs = kwargs.pop("policy_kwargs", {})
        policy_kwargs.setdefault("features_extractor_class", MHACFeaturesExtractor)
        policy_kwargs.setdefault("features_extractor_kwargs", {"latent_dim": 256})
        policy_kwargs.setdefault("net_arch", [])   # no extra MLP on top of encoder
        kwargs["policy_kwargs"] = policy_kwargs

        super().__init__(*args, **kwargs)

        self.predictor = predictor
        self.lambda_pred = lambda_pred
        self.lambda_cons = lambda_cons
        self.horizon = horizon

    def _setup_model(self) -> None:
        super()._setup_model()
        # Policy (and its optimizer) now exist — safe to add predictor params
        if self.predictor is not None:
            self.predictor = self.predictor.to(self.device)
            self.policy.optimizer.add_param_group(
                {"params": self.predictor.parameters()}
            )

    # ------------------------------------------------------------------
    # Aux loss pass (Phases 3–6)
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Standard PPO update, then an optional aux loss pass."""
        super().train()

        if self.predictor is None or (self.lambda_pred == 0.0 and self.lambda_cons == 0.0):
            return

        self._aux_loss_pass()

    def _aux_loss_pass(self) -> None:
        """
        Extract sequential (z_t, a_{t..t+K-1}, z_{t+1..t+K}) tuples from the
        rollout buffer and compute L_pred + L_cons.

        SB3 shuffles the buffer for PPO mini-batches, so we build the
        sequential tuples separately here before the buffer is cleared.
        """
        K = self.horizon
        buf = self.rollout_buffer

        # obs shape: (n_steps, n_envs, *obs_shape)
        obs_np = buf.observations          # (T, n_envs, C, H, W)
        acts_np = buf.actions              # (T, n_envs, 1) or (T, n_envs)
        T, n_envs = obs_np.shape[:2]

        if T < K + 1:
            return

        # Encode all observations at once: (T*n_envs, latent_dim)
        obs_flat = obs_np.reshape(-1, *obs_np.shape[2:])
        obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            z_all = self.policy.features_extractor(obs_t)   # (T*n_envs, latent_dim)
        z_all = z_all.reshape(T, n_envs, -1)                # (T, n_envs, latent_dim)

        acts_flat = acts_np.reshape(T, n_envs).astype(np.int64)  # (T, n_envs)

        # Build (z_t, actions[t..t+K-1], z_targets[t+1..t+K]) for each valid t
        z_t_list, act_seq_list, z_tgt_list = [], [], []
        for t in range(T - K):
            for e in range(n_envs):
                z_t_list.append(z_all[t, e])
                act_seq_list.append(
                    torch.tensor(acts_flat[t:t+K, e], device=self.device)
                )
                z_tgt_list.append(z_all[t+1:t+K+1, e])     # (K, latent_dim)

        if not z_t_list:
            return

        z_t = torch.stack(z_t_list)           # (B, latent_dim)
        action_seq = torch.stack(act_seq_list) # (B, K)
        z_targets = torch.stack(z_tgt_list)   # (B, K, latent_dim)

        # Forward
        z_hat_direct = self.predictor.forward_all_horizons(z_t, action_seq)

        aux_loss = z_t.new_tensor(0.0)

        if self.lambda_pred > 0.0:
            l_pred, _ = prediction_loss(z_hat_direct, z_targets)
            aux_loss = aux_loss + self.lambda_pred * l_pred
            self.logger.record("aux/loss_pred", l_pred.item())

        if self.lambda_cons > 0.0:
            z_hat_chain = self.predictor.chain_with_grad(z_t, action_seq)
            l_cons, _ = consistency_loss(z_hat_chain, z_hat_direct)
            aux_loss = aux_loss + self.lambda_cons * l_cons
            self.logger.record("aux/loss_cons", l_cons.item())

        self.policy.optimizer.zero_grad()
        aux_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.predictor.parameters()),
            self.max_grad_norm,
        )
        self.policy.optimizer.step()
