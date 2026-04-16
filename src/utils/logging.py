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

import os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

# Horizons used when saving latent snapshots during training.
SNAPSHOT_HORIZONS = [1, 3, 5]


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

        # Pull aux losses and drift curve from the trainer's stash.
        # (SB3's logger.dump() clears name_to_value, so we can't read from
        # name_to_value here — the trainer writes to _latest_aux instead.)
        log.update(getattr(self.model, "_latest_aux", {}))

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


# ---------------------------------------------------------------------------
# Latent snapshot collection (shared by callback and smoketest)
# ---------------------------------------------------------------------------

def collect_latent_snapshot(
    model,
    env,
    n_episodes: int,
    horizons: list[int],
) -> dict[str, np.ndarray] | None:
    """
    Roll out *n_episodes* on *env* using *model*, capture encoder outputs, and
    return a dict of numpy arrays suitable for ``np.savez_compressed``.

    Keys always present
    -------------------
    z_t               float32  [N, D]      — encoder output at step t
    actions           int32    [N, max_k]  — action sequence of length max_k
    z_actual_k{k}     float32  [N, D]      — true encoder output k steps later

    Keys present when model.predictor is not None
    ---------------------------------------------
    z_hat_k{k}        float32  [N, D]      — direct k-step prediction

    N = total_steps − max_k (valid starting points).  Returns None when the
    trajectory is too short to form at least one valid window.
    """
    max_k = max(horizons)
    device = model.device

    latents_list: list[np.ndarray] = []
    actions_list: list[int] = []

    captured: dict = {}

    def _hook(module, inp, out):
        captured["z"] = out.detach().cpu()

    hook = model.policy.features_extractor.register_forward_hook(_hook)
    model.policy.set_training_mode(False)

    try:
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                obs_t = torch.as_tensor(
                    np.array([obs]), dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    action, _, _ = model.policy(obs_t)
                latents_list.append(captured["z"].numpy()[0])
                a = int(action.cpu().numpy()[0])
                actions_list.append(a)
                obs, _, term, trunc, _ = env.step(a)
                done = term or trunc
    finally:
        hook.remove()
        model.policy.set_training_mode(True)

    z_all = np.array(latents_list, dtype=np.float32)   # [T, D]
    a_all = np.array(actions_list, dtype=np.int32)      # [T]
    T = len(z_all)

    if T <= max_k:
        return None

    N = T - max_k  # number of valid windows

    z_t = z_all[:N]                                          # [N, D]
    act_seqs = np.stack(
        [a_all[t : t + max_k] for t in range(N)], axis=0
    ).astype(np.int32)                                       # [N, max_k]

    out: dict[str, np.ndarray] = {"z_t": z_t, "actions": act_seqs}

    for k in horizons:
        out[f"z_actual_k{k}"] = z_all[k : k + N]            # [N, D]

    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        z_t_t = torch.as_tensor(z_t, dtype=torch.float32, device=device)
        acts_t = torch.as_tensor(act_seqs.astype(np.int64), dtype=torch.long, device=device)
        with torch.no_grad():
            z_hat_all = predictor.forward_all_horizons(z_t_t, acts_t)  # [N, max_k, D]
        z_hat_np = z_hat_all.cpu().numpy()
        for k in horizons:
            out[f"z_hat_k{k}"] = z_hat_np[:, k - 1, :]     # [N, D]

    return out


# ---------------------------------------------------------------------------
# LatentSnapshotCallback
# ---------------------------------------------------------------------------

class LatentSnapshotCallback(BaseCallback):
    """
    Saves latent snapshots at k=1,3,5 every checkpoint.

    At *snapshot_freq* env steps (measured in ``n_calls``, after dividing by
    ``n_envs`` in the caller, matching how ``CheckpointCallback`` works) this
    callback runs *n_episodes* rollouts on *eval_env*, captures the encoder
    output, and writes an NPZ archive:

        <snapshot_dir>/<run_name>/step_<NNNNNNNNN>.npz

    See ``collect_latent_snapshot`` for the exact array layout.
    """

    def __init__(
        self,
        eval_env,
        snapshot_freq: int = 25_000,
        snapshot_dir: str = "latent_snapshots",
        run_name: str = "run",
        n_episodes: int = 5,
        horizons: list[int] | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.snapshot_freq = snapshot_freq
        self.snapshot_dir = snapshot_dir
        self.run_name = run_name
        self.n_episodes = n_episodes
        self.horizons = sorted(horizons or SNAPSHOT_HORIZONS)
        self._last_snapshot_step: int = -1

    def _on_step(self) -> bool:
        if self.n_calls % self.snapshot_freq == 0:
            self._save_snapshot()
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps != self._last_snapshot_step:
            self._save_snapshot()

    def _save_snapshot(self) -> None:
        step = self.num_timesteps
        self._last_snapshot_step = step

        data = collect_latent_snapshot(
            model=self.model,
            env=self.eval_env,
            n_episodes=self.n_episodes,
            horizons=self.horizons,
        )

        if data is None:
            if self.verbose:
                print(f"[LatentSnapshot] step={step}: trajectory too short, skipping")
            return

        out_dir = Path(self.snapshot_dir) / self.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"step_{step:09d}.npz"
        np.savez_compressed(str(out_path), **data)

        if self.verbose:
            N, D = data["z_t"].shape
            print(
                f"[LatentSnapshot] step={step:>9d}: saved {out_path}  "
                f"N={N}  D={D}  horizons={self.horizons}"
            )
