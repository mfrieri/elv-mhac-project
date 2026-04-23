"""
Policy and value heads that operate on encoder latents.

PolicyHead wraps the actor-critic heads used by SB3's PPO.  It is registered
as the policy_network when subclassing SB3's ActorCriticPolicy so that the
encoder is shared between the RL objective and the auxiliary losses.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .encoder import CNNEncoder
from .miniworld_encoder import MiniWorldCNNEncoder


class MHACFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 features extractor that dispatches to the right encoder by obs shape.

    MiniGrid observations are (3, 7, 7) — use the small CNNEncoder.
    MiniWorld observations are (3, 60, 80) — use the Nature-DQN MiniWorldCNNEncoder.
    """

    def __init__(self, observation_space, latent_dim: int = 256):
        super().__init__(observation_space, features_dim=latent_dim)
        obs_shape = observation_space.shape  # (C, H, W)
        _, h, _ = obs_shape
        if h <= 16:
            self.encoder = CNNEncoder(obs_shape=obs_shape, latent_dim=latent_dim)
        else:
            self.encoder = MiniWorldCNNEncoder(obs_shape=obs_shape, latent_dim=latent_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)


class PolicyHead(nn.Module):
    """
    Thin actor-critic head sitting on top of the encoder latent.

    actor:  latent_dim -> num_actions  (logits)
    critic: latent_dim -> 1            (value estimate)
    """

    def __init__(self, latent_dim: int = 256, num_actions: int = 7):
        super().__init__()
        self.actor = nn.Linear(latent_dim, num_actions)
        self.critic = nn.Linear(latent_dim, 1)

    def forward(self, z: torch.Tensor):
        return self.actor(z), self.critic(z)
