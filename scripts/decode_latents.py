"""
Decode latents to visualize what the agent "imagines" about the world.

For each sampled episode step, produces a figure with three rows:
  Row 1 — Ground truth: actual obs at t, t+1, ..., t+K
  Row 2 — Reconstruction: decoder(z_t) at each step (tests encoder quality)
  Row 3 — Prediction: decoder(z_hat_direct_k) for k=1..K (imagined futures)

Usage:
    python scripts/decode_latents.py \
        --checkpoint checkpoints/fourrooms_mhac_k_seed42/final.zip \
        --condition mhac_k \
        --env fourrooms \
        --n-episodes 10 \
        --output-dir figures/decode
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env.wrappers import get_seed_split, make_env
from src.training.ppo_trainer import MHACTrainer
from src.models.decoder import GridDecoder, N_OBJECT_TYPES

# -------------------------------------------------------------------
# MiniGrid object-type color palette (indices 0-10)
# -------------------------------------------------------------------
OBJECT_NAMES = [
    "unseen", "empty", "wall", "floor", "door",
    "key", "ball", "box", "goal", "lava", "agent",
]

# RGB in [0,1] — one colour per object type
_OBJ_COLORS_RGB = [
    [0.15, 0.15, 0.15],  # 0 unseen   dark grey
    [0.90, 0.90, 0.90],  # 1 empty    light grey
    [0.45, 0.45, 0.45],  # 2 wall     medium grey
    [0.80, 0.80, 0.60],  # 3 floor    beige
    [0.55, 0.35, 0.15],  # 4 door     brown
    [1.00, 0.85, 0.00],  # 5 key      yellow
    [0.00, 0.65, 1.00],  # 6 ball     blue
    [1.00, 0.50, 0.00],  # 7 box      orange
    [0.10, 0.75, 0.10],  # 8 goal     green
    [0.90, 0.10, 0.10],  # 9 lava     red
    [0.70, 0.00, 0.80],  # 10 agent   purple
]
CMAP = ListedColormap(_OBJ_COLORS_RGB)

ENV_NAMES = {
    "fourrooms":    "MiniGrid-FourRooms-v0",
    "multiroom":    "MiniGrid-MultiRoom-N6-v0",
    "multiroom_n4": "MiniGrid-MultiRoom-N4-S5-v0",
}


def obs_to_object_grid(obs_chw: np.ndarray) -> np.ndarray:
    """
    Convert normalized (3, H, W) float32 obs to (H, W) integer object-type array.
    """
    return np.round(obs_chw[0] * 255.0).astype(np.int32)


def render_grid(ax, grid: np.ndarray, title: str = "") -> None:
    """Render a (H, W) integer object-type grid on a matplotlib Axes."""
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=N_OBJECT_TYPES - 1,
              interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=7, pad=2)
    ax.axis("off")


def collect_episode(env, model, decoder, predictor, K: int, device: torch.device):
    """
    Roll out one episode and collect observations, latents, and decoder outputs.

    Returns a list of dicts, one per step:
      obs_chw        : (3, H, W) float32 — raw obs
      z              : (latent_dim,) — encoder output
      grid_recon     : (H, W) int   — decoder(z) argmax
      z_hat_direct   : (K, latent_dim) or None
      grids_pred     : (K, H, W) int or None — decoder(z_hat_direct_k) argmax
    """
    obs, _ = env.reset()
    done = False
    steps = []

    while not done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            z = model.policy.features_extractor(obs_tensor)  # (1, latent_dim)
            grid_recon = decoder.decode_to_grid(z)[0].cpu().numpy()  # (H, W)

            z_hat_direct = None
            grids_pred = None
            if predictor is not None:
                action_tensor = torch.zeros(1, K, dtype=torch.long, device=device)
                action = model.policy.predict(obs, deterministic=True)[0]
                # Fill realistic actions: use the model's policy for k=1, zeros for rest
                action_tensor[0, 0] = int(action)
                z_hat_direct = predictor.forward_all_horizons(z, action_tensor)  # (1, K, D)
                z_hat_direct = z_hat_direct[0]  # (K, D)
                grids_pred = decoder.decode_to_grid(z_hat_direct).cpu().numpy()  # (K, H, W)

        action = model.policy.predict(obs, deterministic=True)[0]
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        steps.append(dict(
            obs_chw=obs_tensor[0].cpu().numpy(),
            z=z[0].cpu().numpy(),
            grid_recon=grid_recon,
            z_hat_direct=z_hat_direct.cpu().numpy() if z_hat_direct is not None else None,
            grids_pred=grids_pred,
        ))

    return steps


def make_legend() -> list:
    return [
        mpatches.Patch(color=_OBJ_COLORS_RGB[i], label=OBJECT_NAMES[i])
        for i in range(N_OBJECT_TYPES)
    ]


def plot_step(step_data: dict, step_idx: int, future_obs: list, K: int, out_path: str) -> None:
    """
    Save a figure for one timestep showing ground truth, reconstruction, and predictions.

    Args:
        step_data : dict from collect_episode for step t
        future_obs: list of (3,H,W) arrays for steps t+1..t+K (may be shorter at episode end)
        K         : prediction horizon
    """
    has_pred = step_data["grids_pred"] is not None
    n_rows = 3 if has_pred else 2
    n_cols = K + 1  # t, t+1, ..., t+K

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.8 * n_rows + 0.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Row 0: ground truth
    render_grid(axes[0, 0], obs_to_object_grid(step_data["obs_chw"]), f"GT t={step_idx}")
    for k in range(1, n_cols):
        if k - 1 < len(future_obs):
            render_grid(axes[0, k], obs_to_object_grid(future_obs[k - 1]), f"GT t+{k}")
        else:
            axes[0, k].axis("off")

    # Row 1: reconstruction of current frame (and blanks for future columns)
    render_grid(axes[1, 0], step_data["grid_recon"], "Recon z_t")
    for k in range(1, n_cols):
        axes[1, k].axis("off")
    axes[1, 0].set_title("Recon z_t", fontsize=7, pad=2)

    # Row 2 (optional): predicted future grids
    if has_pred:
        axes[2, 0].axis("off")
        for k in range(1, n_cols):
            if k - 1 < K:
                render_grid(axes[2, k], step_data["grids_pred"][k - 1], f"Pred k={k}")
            else:
                axes[2, k].axis("off")

    row_labels = ["Ground truth", "Reconstruction", "Prediction"][:n_rows]
    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(
            label, xy=(0, 0.5), xytext=(-0.3, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            fontsize=7, ha="right", va="center", rotation=90,
        )

    legend = make_legend()
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=6,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Step {step_idx}", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--env", required=True, choices=list(ENV_NAMES.keys()))
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--steps-per-episode", type=int, default=3,
                   help="Number of timesteps within each episode to plot")
    p.add_argument("--output-dir", default="figures/decode")
    p.add_argument("--seed-pool", type=int, default=500)
    p.add_argument("--horizon", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = ENV_NAMES[args.env]

    _, test_seeds = get_seed_split(seed_pool=args.seed_pool, test_fraction=0.2, rng_seed=0)
    env = make_env(env_name, test_seeds)

    # Load checkpoint
    model = MHACTrainer.load(args.checkpoint, device=device)
    model.policy.set_training_mode(False)

    # Load decoder — must be trained alongside the checkpoint; rebuild with same dims
    decoder = GridDecoder(latent_dim=256).to(device)
    decoder_path = Path(args.checkpoint).parent / "decoder.pt"
    if decoder_path.exists():
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        print(f"Loaded decoder from {decoder_path}")
    else:
        print(f"[warn] No decoder weights found at {decoder_path}; using random weights.")
    decoder.eval()

    predictor = getattr(model, "predictor", None)
    K = args.horizon

    for ep in range(args.n_episodes):
        steps = collect_episode(env, model, decoder, predictor, K, device)
        n = len(steps)
        if n == 0:
            continue

        # Sample evenly-spaced steps from the episode (avoid last K steps for future GT)
        sample_indices = np.linspace(0, max(0, n - K - 1), args.steps_per_episode, dtype=int)
        sample_indices = list(dict.fromkeys(sample_indices.tolist()))  # deduplicate

        for t in sample_indices:
            future_obs = [steps[t + k]["obs_chw"] for k in range(1, K + 1) if t + k < n]
            out_path = os.path.join(args.output_dir, f"ep{ep:02d}_t{t:04d}.png")
            plot_step(steps[t], t, future_obs, K, out_path)
            print(f"  saved {out_path}")

        print(f"Episode {ep}: {n} steps")

    env.close()
    print(f"\nFigures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
