"""
Latent saving pipeline — 50 test-seed episodes.

Rolls out a trained MHAC checkpoint over test seeds, captures the encoder's
latent representation at every timestep, and writes a structured NPZ archive.

Usage:
    python scripts/save_latents.py \
        --checkpoint checkpoints/fourrooms_mhac_k/final.zip \
        --condition mhac_k \
        --env fourrooms \
        --n-episodes 50 \
        --output-dir latents/

Output (one file per run):
    latents/<env>_<condition>_latents.npz
    Keys
    ----
    latents      : float32  [N_steps, latent_dim]   — encoder output z_t
    actions      : int32    [N_steps]                — action taken at t
    rewards      : float32  [N_steps]                — reward at t
    dones        : bool     [N_steps]                — episode boundary flag
    episode_ids  : int32    [N_steps]                — which episode each step belongs to
    seed_ids     : int32    [N_episodes]             — test seed for episode i
    episode_returns : float32 [N_episodes]           — total undiscounted return per episode
    episode_lengths : int32   [N_episodes]           — steps per episode
    metadata     : (stored as attrs dict in a companion JSON)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env.wrappers import get_seed_split, make_env
from src.training.ppo_trainer import MHACTrainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Save encoder latents over test episodes.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to a saved MHACTrainer .zip checkpoint")
    p.add_argument("--condition", required=True,
                   help="Condition label (for output filename, e.g. mhac_k, baseline)")
    p.add_argument("--env", required=True, choices=["fourrooms", "multiroom"],
                   help="Environment key")
    p.add_argument("--seed-pool", type=int, default=500,
                   help="Must match the pool used during training")
    p.add_argument("--n-episodes", type=int, default=50,
                   help="Number of test episodes to collect")
    p.add_argument("--max-steps-per-episode", type=int, default=500,
                   help="Hard cap on steps per episode (env timeout override)")
    p.add_argument("--output-dir", default="latents",
                   help="Directory to write the .npz and .json files")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for episode ordering")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Torch device for model inference")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

ENV_NAMES = {
    "fourrooms": "MiniGrid-FourRooms-v0",
    "multiroom": "MiniGrid-MultiRoom-N6-v0",
}


# ---------------------------------------------------------------------------
# Latent extraction hook
# ---------------------------------------------------------------------------

class LatentRecorder:
    """
    Registers a forward hook on the policy's mlp_extractor (or features_extractor)
    to capture the encoder output z_t at each forward pass.

    Works with SB3's MlpPolicy: the features_extractor maps obs -> flat features,
    which serves as z_t in our MHAC setup.
    """

    def __init__(self, model: MHACTrainer):
        self.latent: torch.Tensor | None = None
        # SB3 MlpPolicy: features_extractor is the obs encoder
        self._hook = model.policy.features_extractor.register_forward_hook(
            self._hook_fn
        )

    def _hook_fn(self, module, input, output):
        # output shape: [batch, latent_dim] — detach immediately
        self.latent = output.detach().cpu()

    def remove(self):
        self._hook.remove()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args):
    env_name = ENV_NAMES[args.env]
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Reproduce the same test seed split used in training ----
    _, test_seeds = get_seed_split(
        seed_pool=args.seed_pool, test_fraction=0.2, rng_seed=0
    )
    rng = np.random.default_rng(args.seed)
    episode_seeds = rng.choice(test_seeds, size=args.n_episodes, replace=True).tolist()

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Environment: {env_name}")
    print(f"Episodes   : {args.n_episodes}  |  test pool size: {len(test_seeds)}")

    # ---- Load model ----
    # We need a dummy env to satisfy SB3's constructor when loading
    dummy_env = make_env(env_name, test_seeds)
    model = MHACTrainer.load(
        args.checkpoint,
        env=dummy_env,
        device=args.device,
    )
    model.policy.set_training_mode(False)

    recorder = LatentRecorder(model)

    # ---- Collect episodes ----
    all_latents   = []   # [N_steps, latent_dim]
    all_actions   = []   # [N_steps]
    all_rewards   = []   # [N_steps]
    all_dones     = []   # [N_steps]
    all_ep_ids    = []   # [N_steps]

    ep_returns    = []   # [N_episodes]
    ep_lengths    = []   # [N_episodes]
    ep_seeds_used = []   # [N_episodes]

    for ep_idx, seed in enumerate(episode_seeds):
        env = make_env(env_name, [seed])
        obs, _ = env.reset(seed=seed)

        ep_return = 0.0
        ep_len    = 0
        done      = False

        while not done and ep_len < args.max_steps_per_episode:
            obs_tensor = torch.as_tensor(
                np.array([obs]), dtype=torch.float32, device=args.device
            )

            with torch.no_grad():
                action, _, _ = model.policy(obs_tensor)

            action_np = action.cpu().numpy()[0]

            # The hook fires during model.policy(obs_tensor) above,
            # so recorder.latent is already populated.
            latent_np = recorder.latent.numpy()[0]   # [latent_dim]

            obs_next, reward, terminated, truncated, _ = env.step(int(action_np))
            done = terminated or truncated

            all_latents.append(latent_np)
            all_actions.append(int(action_np))
            all_rewards.append(float(reward))
            all_dones.append(bool(done))
            all_ep_ids.append(ep_idx)

            ep_return += float(reward)
            ep_len    += 1
            obs        = obs_next

        env.close()

        ep_returns.append(ep_return)
        ep_lengths.append(ep_len)
        ep_seeds_used.append(seed)

        if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
            print(
                f"  Episode {ep_idx+1:3d}/{args.n_episodes}  "
                f"seed={seed:4d}  return={ep_return:.2f}  len={ep_len}"
            )

    recorder.remove()
    dummy_env.close()

    # ---- Pack arrays ----
    latents_arr   = np.array(all_latents,  dtype=np.float32)   # [T, D]
    actions_arr   = np.array(all_actions,  dtype=np.int32)     # [T]
    rewards_arr   = np.array(all_rewards,  dtype=np.float32)   # [T]
    dones_arr     = np.array(all_dones,    dtype=bool)         # [T]
    ep_ids_arr    = np.array(all_ep_ids,   dtype=np.int32)     # [T]
    ep_returns_arr= np.array(ep_returns,   dtype=np.float32)   # [E]
    ep_lengths_arr= np.array(ep_lengths,   dtype=np.int32)     # [E]
    ep_seeds_arr  = np.array(ep_seeds_used,dtype=np.int32)     # [E]

    # ---- Save NPZ ----
    stem    = f"{args.env}_{args.condition}_latents"
    npz_path = os.path.join(args.output_dir, stem + ".npz")
    np.savez_compressed(
        npz_path,
        latents        = latents_arr,
        actions        = actions_arr,
        rewards        = rewards_arr,
        dones          = dones_arr,
        episode_ids    = ep_ids_arr,
        seed_ids       = ep_seeds_arr,
        episode_returns= ep_returns_arr,
        episode_lengths= ep_lengths_arr,
    )

    # ---- Save companion metadata JSON ----
    metadata = dict(
        checkpoint      = str(args.checkpoint),
        condition       = args.condition,
        env             = args.env,
        env_name        = env_name,
        n_episodes      = args.n_episodes,
        total_steps     = int(latents_arr.shape[0]),
        latent_dim      = int(latents_arr.shape[1]),
        seed_pool       = args.seed_pool,
        collection_seed = args.seed,
        mean_return     = float(ep_returns_arr.mean()),
        std_return      = float(ep_returns_arr.std()),
        mean_length     = float(ep_lengths_arr.mean()),
        success_rate    = float((ep_returns_arr > 0).mean()),
    )
    json_path = os.path.join(args.output_dir, stem + "_meta.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # ---- Summary ----
    print("\n── Collection complete ──────────────────────────────────────────")
    print(f"  Total steps  : {latents_arr.shape[0]}")
    print(f"  Latent dim   : {latents_arr.shape[1]}")
    print(f"  Mean return  : {ep_returns_arr.mean():.3f} ± {ep_returns_arr.std():.3f}")
    print(f"  Success rate : {metadata['success_rate']:.1%}")
    print(f"  Mean ep len  : {ep_lengths_arr.mean():.1f}")
    print(f"\n  Saved → {npz_path}")
    print(f"  Meta  → {json_path}")
    print("────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    args = parse_args()
    run(args)