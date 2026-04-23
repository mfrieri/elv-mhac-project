"""
MiniWorld environment wrappers and factories.

Kept separate from the MiniGrid pipeline in src/env/wrappers.py. The only
shared pieces are the seed-split helper and the SeededEnv wrapper, which
are imported from that module so the 80/20 train/test split stays
identical across both environment families.

All wrappers produce observations as (C, H, W) float32 tensors normalised to [0, 1].
"""

import gymnasium as gym
import miniworld  # noqa: F401 — registers MiniWorld envs with gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from .wrappers import SeededEnv, get_seed_split  # noqa: F401 — re-exported


# ---------------------------------------------------------------------------
# Observation wrapper
# ---------------------------------------------------------------------------

class RGBToCHW(gym.ObservationWrapper):
    """Convert MiniWorld (H, W, 3) uint8 image to (3, H, W) float32 in [0, 1]."""

    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32
        )

    def observation(self, obs):
        img = obs.astype(np.float32) / 255.0   # (H, W, C)
        return np.transpose(img, (2, 0, 1))    # (C, H, W)


# ---------------------------------------------------------------------------
# Action-space restriction
# ---------------------------------------------------------------------------

# MiniWorld's base Actions enum: 0=turn_left, 1=turn_right, 2=move_forward,
# 3=move_back, 4=pickup, 5=drop, 6=toggle, 7=done. Navigation envs
# (Hallway, OneRoom, FourRooms) only need 0..2; the rest are no-ops and
# just waste exploration budget.

class NavActions(gym.ActionWrapper):
    """Restrict to {turn_left, turn_right, move_forward} as Discrete(3)."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        return int(action)  # already aligned with native action ids 0/1/2


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_single_env(env_name: str, seeds: list, restrict_actions: bool = True):
    def _init():
        env = gym.make(env_name)
        env = SeededEnv(env, seeds)
        if restrict_actions:
            env = NavActions(env)
        env = RGBToCHW(env)
        return env
    return _init


def make_vec_envs(
    env_name: str,
    seeds: list,
    num_envs: int = 4,
    restrict_actions: bool = True,
) -> VecEnv:
    """Create a vectorised set of MiniWorld training environments."""
    return make_vec_env(
        _make_single_env(env_name, seeds, restrict_actions=restrict_actions),
        n_envs=num_envs,
    )


def make_env(
    env_name: str,
    seeds: list,
    restrict_actions: bool = True,
) -> gym.Env:
    """Create a single (non-vectorised) MiniWorld environment for evaluation."""
    return _make_single_env(env_name, seeds, restrict_actions=restrict_actions)()
