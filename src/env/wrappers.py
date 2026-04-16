"""
Environment wrappers and seed-split utilities.

The train/test split is defined once here using a fixed pool of integer seeds and
never changed.  80% of seeds go to training, 20% to the held-out test set.

All wrappers produce observations as (C, H, W) float32 tensors normalised to [0, 1].
"""

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv


# ---------------------------------------------------------------------------
# Seed split
# ---------------------------------------------------------------------------

def get_seed_split(seed_pool: int = 500, test_fraction: float = 0.2, rng_seed: int = 0):
    """Return (train_seeds, test_seeds) as lists of ints.

    The split is deterministic given the arguments and must never change once
    training has started on a set of experiments.
    """
    rng = np.random.default_rng(rng_seed)
    all_seeds = np.arange(seed_pool)
    rng.shuffle(all_seeds)
    n_test = int(seed_pool * test_fraction)
    test_seeds = all_seeds[:n_test].tolist()
    train_seeds = all_seeds[n_test:].tolist()
    return train_seeds, test_seeds


# ---------------------------------------------------------------------------
# Observation wrapper
# ---------------------------------------------------------------------------

class ImageToCHW(gym.ObservationWrapper):
    """Convert MiniGrid (H, W, C) uint8 image to (C, H, W) float32 in [0, 1]."""

    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        # MiniGrid wraps obs in a dict with key "image"
        img_space = obs_space["image"]
        h, w, c = img_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32
        )

    def observation(self, obs):
        img = obs["image"].astype(np.float32) / 255.0  # (H, W, C)
        return np.transpose(img, (2, 0, 1))             # (C, H, W)


class SeededEnv(gym.Wrapper):
    """Sample a random seed from a fixed pool at each reset."""

    def __init__(self, env, seeds: list):
        super().__init__(env)
        self._seeds = seeds
        self._rng = np.random.default_rng()

    def reset(self, **kwargs):
        kwargs.pop("seed", None)  # ignore SB3's seed; we control it from our pool
        seed = int(self._rng.choice(self._seeds))
        return self.env.reset(seed=seed, **kwargs)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

MAX_STEPS_OVERRIDE = {
    "MiniGrid-MultiRoom-N4-S5-v0": 3_000,
    "MiniGrid-MultiRoom-N6-v0":    10_000,
}


def _make_single_env(env_name: str, seeds: list):
    def _init():
        kwargs = {}
        if env_name in MAX_STEPS_OVERRIDE:
            kwargs["max_steps"] = MAX_STEPS_OVERRIDE[env_name]
        env = gym.make(env_name, **kwargs)
        env = SeededEnv(env, seeds)
        env = ImageToCHW(env)
        return env
    return _init


def make_vec_envs(env_name: str, seeds: list, num_envs: int = 4) -> VecEnv:
    """Create a vectorised set of training environments."""
    return make_vec_env(
        _make_single_env(env_name, seeds),
        n_envs=num_envs,
    )


def make_env(env_name: str, seeds: list) -> gym.Env:
    """Create a single (non-vectorised) environment for evaluation."""
    return _make_single_env(env_name, seeds)()
