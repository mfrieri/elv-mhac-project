"""
W&B logging callback for MHAC training.

Logs at a fixed evaluation interval (every eval_freq timesteps):
  eval/train_sr          — success rate on training seeds
  eval/test_sr           — success rate on held-out test seeds
  eval/train_reward      — mean episode reward on training seeds
  eval/test_reward       — mean episode reward on test seeds
  eval/gen_gap           — train SR minus test SR
  aux/loss_pred          — L_pred (when applicable)
  aux/loss_cons          — L_cons (when applicable)
  aux/pred_k{k}          — per-horizon prediction error (when applicable)
"""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class WandBEvalCallback(BaseCallback):
    """
    SB3 callback that periodically evaluates the policy on train and test
    seeds and logs everything to W&B.
    """

    def __init__(
        self,
        train_env,
        test_env,
        n_eval_episodes: int = 50,
        eval_freq: int = 50_000,
        run_name: str = "mhac",
        condition: str = "baseline",
        env_name: str = "fourrooms",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.train_env = train_env
        self.test_env = test_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.run_name = run_name
        self.condition = condition
        self.env_name = env_name

    def _on_training_start(self) -> None:
        import wandb
        wandb.init(
            project="mhac",
            name=self.run_name,
            config={
                "condition": self.condition,
                "env": self.env_name,
            },
            resume="allow",
        )

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            self._run_eval()
        return True

    def _on_training_end(self) -> None:
        import wandb
        self._run_eval()
        wandb.finish()

    def _run_eval(self) -> None:
        import wandb

        train_metrics = self._evaluate(self.train_env)
        test_metrics  = self._evaluate(self.test_env)

        log = {
            "timestep":           self.num_timesteps,
            "eval/train_sr":      train_metrics["success_rate"],
            "eval/test_sr":       test_metrics["success_rate"],
            "eval/train_reward":  train_metrics["mean_reward"],
            "eval/test_reward":   test_metrics["mean_reward"],
            "eval/gen_gap":       train_metrics["success_rate"] - test_metrics["success_rate"],
        }

        # Pull aux losses out of the SB3 logger if they were recorded
        for key in ("aux/loss_pred", "aux/loss_cons"):
            val = self._get_logged_value(key)
            if val is not None:
                log[key] = val

        wandb.log(log, step=self.num_timesteps)

        if self.verbose:
            print(
                f"[{self.num_timesteps:>8d}] "
                f"train_sr={train_metrics['success_rate']:.3f}  "
                f"test_sr={test_metrics['success_rate']:.3f}  "
                f"gen_gap={log['eval/gen_gap']:.3f}"
            )

    def _evaluate(self, env) -> dict:
        """Roll out n_eval_episodes and return SR + mean reward."""
        policy = self.model
        episode_rewards = []
        successes = 0

        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = policy.predict(
                    obs[np.newaxis], deterministic=True
                )
                obs, reward, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
                ep_reward += float(reward)
            episode_rewards.append(ep_reward)
            if ep_reward > 0:
                successes += 1

        return {
            "success_rate": successes / self.n_eval_episodes,
            "mean_reward":  float(np.mean(episode_rewards)),
        }

    def _get_logged_value(self, key: str):
        """Pull a scalar out of SB3's internal logger if it exists."""
        try:
            return self.logger.name_to_value.get(key)
        except Exception:
            return None
