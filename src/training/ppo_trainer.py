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
from src.training.losses import prediction_loss, consistency_loss, reconstruction_loss
from src.evaluation.diagnostics import drift_curve


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
        decoder=None,
        lambda_recon: float = 0.0,
        **kwargs,
    ):
        # Inject our feature extractor so the CNN encoder is used automatically
        policy_kwargs = kwargs.pop("policy_kwargs", {})
        policy_kwargs.setdefault("features_extractor_class", MHACFeaturesExtractor)
        policy_kwargs.setdefault("features_extractor_kwargs", {"latent_dim": 256})
        policy_kwargs.setdefault("net_arch", [])   # no extra MLP on top of encoder
        kwargs["policy_kwargs"] = policy_kwargs

        # Set these before super().__init__() because SB3 calls _setup_model()
        # inside __init__, which references these attributes.
        self.predictor = predictor
        self.lambda_pred = lambda_pred
        self.lambda_cons = lambda_cons
        self.horizon = horizon
        self.decoder = decoder
        self.lambda_recon = lambda_recon

        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        # Policy (and its optimizer) now exist — safe to add predictor params.
        # Use getattr because SB3's load() bypasses __init__ and calls this directly.
        if getattr(self, "predictor", None) is not None:
            self.predictor = self.predictor.to(self.device)
            self.policy.optimizer.add_param_group(
                {"params": self.predictor.parameters()}
            )
        if getattr(self, "decoder", None) is not None:
            self.decoder = self.decoder.to(self.device)
            self.policy.optimizer.add_param_group(
                {"params": self.decoder.parameters()}
            )

    # ------------------------------------------------------------------
    # Aux loss pass (Phases 3–6)
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Standard PPO update, then an optional aux loss pass."""
        super().train()

        has_pred = self.predictor is not None and (self.lambda_pred > 0.0 or self.lambda_cons > 0.0)
        has_recon = self.decoder is not None and self.lambda_recon > 0.0
        if not has_pred and not has_recon:
            return

        self._aux_loss_pass()

    def _aux_loss_pass(self) -> None:
        """
        Compute auxiliary losses over the current rollout buffer:
          - L_pred: multi-horizon direct prediction loss
          - L_cons: autoregressive consistency loss
          - L_recon: grid reconstruction loss (encoder trained to preserve spatial structure)

        SB3 shuffles the buffer for PPO mini-batches, so sequential tuples are
        built here separately before the buffer is cleared.
        """
        K = self.horizon
        buf = self.rollout_buffer
        obs_np = buf.observations
        T, n_envs = buf.buffer_size, buf.n_envs

        obs_shape = self.observation_space.shape
        obs_flat = obs_np.reshape(T * n_envs, *obs_shape)
        obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=self.device)

        self._latest_aux = {}
        aux_loss = obs_t.new_tensor(0.0)

        # ------------------------------------------------------------------
        # Reconstruction loss (decoder only, re-encodes with gradients)
        # ------------------------------------------------------------------
        if self.decoder is not None and self.lambda_recon > 0.0:
            z_for_recon = self.policy.features_extractor(obs_t)   # grad flows to encoder
            recon_logits = self.decoder(z_for_recon)
            l_recon = reconstruction_loss(recon_logits, obs_t)
            aux_loss = aux_loss + self.lambda_recon * l_recon
            self.logger.record("aux/loss_recon", l_recon.item())
            self._latest_aux["aux/loss_recon"] = l_recon.item()

        # ------------------------------------------------------------------
        # Prediction + consistency losses (require sequential tuples)
        # ------------------------------------------------------------------
        has_pred = self.predictor is not None and (self.lambda_pred > 0.0 or self.lambda_cons > 0.0)
        if has_pred:
            if T < K + 1:
                if aux_loss.requires_grad:
                    self.policy.optimizer.zero_grad()
                    aux_loss.backward()
                    self.policy.optimizer.step()
                    self.logger.dump(step=self.num_timesteps)
                return

            with torch.no_grad():
                z_all = self.policy.features_extractor(obs_t)   # (T*n_envs, latent_dim)
            z_all = z_all.reshape(T, n_envs, -1)

            acts_np = buf.actions
            acts_flat = acts_np.reshape(T, n_envs).astype(np.int64)

            z_t_list, act_seq_list, z_tgt_list = [], [], []
            for t in range(T - K):
                for e in range(n_envs):
                    z_t_list.append(z_all[t, e])
                    act_seq_list.append(
                        torch.tensor(acts_flat[t:t+K, e], device=self.device)
                    )
                    z_tgt_list.append(z_all[t+1:t+K+1, e])

            if z_t_list:
                z_t = torch.stack(z_t_list)
                action_seq = torch.stack(act_seq_list)
                z_targets = torch.stack(z_tgt_list)

                z_hat_direct = self.predictor.forward_all_horizons(z_t, action_seq)

                z_hat_chain_eval = self.predictor.chain(z_t, action_seq)
                drift = drift_curve(z_hat_direct.detach(), z_hat_chain_eval)
                for k, dk in enumerate(drift, start=1):
                    self.logger.record(f"aux/drift_k{k}", dk.item())
                self._latest_aux.update({f"aux/drift_k{k}": dk.item()
                                         for k, dk in enumerate(drift, start=1)})

                if self.lambda_pred > 0.0:
                    l_pred, per_horizon = prediction_loss(z_hat_direct, z_targets)
                    aux_loss = aux_loss + self.lambda_pred * l_pred
                    self.logger.record("aux/loss_pred", l_pred.item())
                    for k, lk in enumerate(per_horizon, start=1):
                        self.logger.record(f"aux/loss_pred_k{k}", lk.item())
                    self._latest_aux["aux/loss_pred"] = l_pred.item()
                    self._latest_aux.update({f"aux/loss_pred_k{k}": lk.item()
                                             for k, lk in enumerate(per_horizon, start=1)})

                if self.lambda_cons > 0.0:
                    z_hat_chain = self.predictor.chain_with_grad(z_t, action_seq)
                    l_cons, _ = consistency_loss(z_hat_chain, z_hat_direct)
                    aux_loss = aux_loss + self.lambda_cons * l_cons
                    self.logger.record("aux/loss_cons", l_cons.item())
                    self._latest_aux["aux/loss_cons"] = l_cons.item()

        self.policy.optimizer.zero_grad()
        aux_loss.backward()
        clip_params = list(self.policy.parameters())
        if self.predictor is not None:
            clip_params += list(self.predictor.parameters())
        if self.decoder is not None:
            clip_params += list(self.decoder.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, self.max_grad_norm)
        self.policy.optimizer.step()
        self.logger.dump(step=self.num_timesteps)
