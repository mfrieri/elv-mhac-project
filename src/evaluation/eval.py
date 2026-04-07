"""
Policy evaluation utilities.

evaluate_policy() runs n_episodes of rollouts on a single (non-vectorised)
environment and returns success rate and mean episode reward.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch


def evaluate_policy(
    policy,
    encoder,
    env: gym.Env,
    n_episodes: int = 50,
    device: str = "cpu",
) -> dict:
    """
    Evaluate a policy for n_episodes and return summary statistics.

    Args:
        policy:     callable(z) -> action  (e.g. the SB3 policy's predict method)
        encoder:    CNNEncoder module
        env:        gymnasium environment (single, not vectorised)
        n_episodes: number of evaluation episodes
        device:     torch device string

    Returns:
        dict with keys:
            success_rate   — fraction of episodes that reached the goal
            mean_reward    — mean cumulative reward per episode
            episode_rewards — list of per-episode cumulative rewards
    """
    encoder.eval()
    episode_rewards = []
    successes = 0

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                z = encoder(obs_t)
                action, _ = policy.predict(obs_t.cpu().numpy(), deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
                ep_reward += float(reward)

            episode_rewards.append(ep_reward)
            # MiniGrid: positive reward only on success
            if ep_reward > 0:
                successes += 1

    encoder.train()
    return {
        "success_rate": successes / n_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "episode_rewards": episode_rewards,
    }
