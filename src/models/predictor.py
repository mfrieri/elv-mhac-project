"""
MHAC Transformer Predictor.

Supports two call modes via the same weights:

  Direct (k-step in one pass):
      forward(z_t, actions)  where actions has shape (batch, k)
      Returns z_hat_direct of shape (batch, latent_dim).

  Chained (autoregressive, evaluation only):
      chain(z_t, actions)  where actions has shape (batch, K)
      Returns z_hat_chain of shape (batch, K, latent_dim).

Architecture
------------
Input sequence: [z_t token, a_t embed, a_{t+1} embed, ..., a_{t+k-1} embed, PREDICT token]
                 length = k + 2  (latent + k actions + predict token)

Transformer output at the [PREDICT] position is projected to latent_dim.

Hyperparameters (from plan):
  num_layers = 2
  num_heads  = 4
  d_model    = latent_dim
"""

import torch
import torch.nn as nn


class MHACPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        num_actions: int = 7,     # MiniGrid default action count
        num_layers: int = 2,
        num_heads: int = 4,
        use_action_conditioning: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_action_conditioning = use_action_conditioning

        # Project latent into transformer d_model (same dimension)
        self.latent_proj = nn.Linear(latent_dim, latent_dim)

        # Action embedding (one-hot -> latent_dim)
        self.action_embed = nn.Embedding(num_actions, latent_dim)

        # Learned [PREDICT] token
        self.predict_token = nn.Parameter(torch.randn(1, 1, latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project transformer output at [PREDICT] position -> latent_dim
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sequence(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Build transformer input sequence for a single horizon.

        Args:
            z:       (batch, latent_dim)
            actions: (batch, k) long — action indices for k steps
        Returns:
            seq: (batch, k+2, latent_dim)
        """
        batch = z.shape[0]

        # Latent token: (batch, 1, latent_dim)
        z_tok = self.latent_proj(z).unsqueeze(1)

        if self.use_action_conditioning:
            # Action tokens: (batch, k, latent_dim)
            a_toks = self.action_embed(actions)
        else:
            # Zero out action conditioning (Condition 6)
            a_toks = torch.zeros(
                batch, actions.shape[1], self.latent_dim, device=z.device
            )

        # [PREDICT] token: (batch, 1, latent_dim)
        pred_tok = self.predict_token.expand(batch, -1, -1)

        return torch.cat([z_tok, a_toks, pred_tok], dim=1)  # (batch, k+2, latent_dim)

    def _predict_from_seq(self, seq: torch.Tensor) -> torch.Tensor:
        """Run transformer and extract output at [PREDICT] position."""
        out = self.transformer(seq)       # (batch, k+2, latent_dim)
        return self.output_proj(out[:, -1, :])  # (batch, latent_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Direct k-step prediction in a single transformer forward pass.

        Args:
            z:       (batch, latent_dim)  — z_t
            actions: (batch, k) long      — a_t, ..., a_{t+k-1}
        Returns:
            z_hat_direct: (batch, latent_dim)
        """
        seq = self._build_sequence(z, actions)
        return self._predict_from_seq(seq)

    def forward_all_horizons(
        self, z: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Direct predictions for all horizons k = 1, ..., K simultaneously.

        Implements vectorised horizon batching: folds (batch, K) into a single
        transformer call by broadcasting over horizons.

        Args:
            z:       (batch, latent_dim)
            actions: (batch, K) long
        Returns:
            z_hat_direct: (batch, K, latent_dim)
        """
        batch, K = actions.shape
        # Build sequences for all horizons — (batch*K, k+2, latent_dim)
        seqs = []
        for k in range(1, K + 1):
            seq = self._build_sequence(z, actions[:, :k])  # (batch, k+2, latent_dim)
            # Pad to length K+2 so we can stack uniformly
            pad_len = K - k
            if pad_len > 0:
                pad = torch.zeros(
                    batch, pad_len, self.latent_dim, device=z.device
                )
                seq = torch.cat([seq, pad], dim=1)
            seqs.append(seq)

        # Stack: (K, batch, K+2, latent_dim) -> (batch*K, K+2, latent_dim)
        stacked = torch.stack(seqs, dim=0)          # (K, batch, K+2, latent_dim)
        stacked = stacked.view(K * batch, K + 2, self.latent_dim)

        out = self.transformer(stacked)             # (K*batch, K+2, latent_dim)
        preds = self.output_proj(out[:, -1, :])     # (K*batch, latent_dim)
        return preds.view(K, batch, self.latent_dim).permute(1, 0, 2)
        # returns (batch, K, latent_dim)

    @torch.no_grad()
    def chain(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive chained prediction (evaluation only, no gradients).

        Feeds each predicted latent back as z_{t+k} for the next step.

        Args:
            z:       (batch, latent_dim)  — z_t
            actions: (batch, K) long
        Returns:
            z_hat_chain: (batch, K, latent_dim)
        """
        batch, K = actions.shape
        results = []
        z_curr = z
        for k in range(K):
            a_k = actions[:, k:k+1]                    # (batch, 1)
            z_next = self.forward(z_curr, a_k)          # (batch, latent_dim)
            results.append(z_next)
            z_curr = z_next
        return torch.stack(results, dim=1)              # (batch, K, latent_dim)

    def chain_with_grad(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive chained prediction with gradients (used in L_cons).

        Same as chain() but allows gradient flow through the chained rollout
        so that L_cons can update the predictor weights.
        """
        batch, K = actions.shape
        results = []
        z_curr = z
        for k in range(K):
            a_k = actions[:, k:k+1]
            z_next = self.forward(z_curr, a_k)
            results.append(z_next)
            z_curr = z_next
        return torch.stack(results, dim=1)              # (batch, K, latent_dim)
