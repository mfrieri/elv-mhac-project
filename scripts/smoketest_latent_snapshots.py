"""
Smoketest for latent snapshot collection.

Two modes:

1. Generate-and-verify (default, no checkpoint needed):
   Creates a freshly-initialised MHACTrainer + predictor, runs
   collect_latent_snapshot() directly, and checks that all expected keys and
   shapes are correct for both the baseline (no predictor) and mhac_k
   (with predictor) cases.

       python scripts/smoketest_latent_snapshots.py

2. Verify an existing snapshot directory:
   Loads every .npz in the directory and checks structure / shape consistency.

       python scripts/smoketest_latent_snapshots.py --snapshot-dir latent_snapshots/fourrooms_mhac_k/

Exit code 0 = pass, non-zero = failure.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HORIZONS = [1, 3, 5]
LATENT_DIM = 64   # small for fast smoke test
N_EPISODES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)


def _check_snapshot_dict(data: dict, has_predictor: bool, horizons: list[int]) -> None:
    """Verify keys, dtypes, and shape consistency of a snapshot dict."""
    N, D = data["z_t"].shape
    max_k = max(horizons)

    _assert(data["z_t"].dtype.kind == "f", "z_t must be float")
    _assert(data["actions"].shape == (N, max_k), f"actions shape wrong: {data['actions'].shape}")
    _assert(data["actions"].dtype.kind in ("i", "u"), "actions must be int")

    for k in horizons:
        key_actual = f"z_actual_k{k}"
        _assert(key_actual in data, f"missing key {key_actual}")
        _assert(data[key_actual].shape == (N, D),
                f"{key_actual} shape {data[key_actual].shape} != ({N},{D})")

        if has_predictor:
            key_hat = f"z_hat_k{k}"
            _assert(key_hat in data, f"missing key {key_hat}")
            _assert(data[key_hat].shape == (N, D),
                    f"{key_hat} shape {data[key_hat].shape} != ({N},{D})")

    print(f"  ok  N={N}  D={D}  horizons={horizons}  predictor={has_predictor}")


# ---------------------------------------------------------------------------
# Mode 1: generate from a fresh model
# ---------------------------------------------------------------------------

def _run_generate_mode() -> None:
    import gymnasium as gym
    import minigrid  # noqa: F401 — registers envs
    import torch

    from src.env.wrappers import get_seed_split, make_env
    from src.models.predictor import MHACPredictor
    from src.training.ppo_trainer import MHACTrainer
    from src.utils.logging import collect_latent_snapshot

    env_name = "MiniGrid-FourRooms-v0"
    _, test_seeds = get_seed_split(seed_pool=500, test_fraction=0.2, rng_seed=0)

    sample_env = gym.make(env_name)
    num_actions = sample_env.action_space.n
    sample_env.close()

    # ---- Case 1: baseline (no predictor) --------------------------------
    print("\n[1/2] baseline (no predictor)")
    env_base = make_env(env_name, test_seeds)
    model_base = MHACTrainer(
        policy="MlpPolicy",
        env=env_base,
        policy_kwargs={
            "features_extractor_kwargs": {"latent_dim": LATENT_DIM},
            "net_arch": [],
        },
        n_steps=32,
        batch_size=32,
        verbose=0,
    )
    eval_env_base = make_env(env_name, test_seeds)
    data_base = collect_latent_snapshot(
        model=model_base,
        env=eval_env_base,
        n_episodes=N_EPISODES,
        horizons=HORIZONS,
    )
    _assert(data_base is not None, "collect returned None for baseline")
    _check_snapshot_dict(data_base, has_predictor=False, horizons=HORIZONS)
    env_base.close()
    eval_env_base.close()

    # ---- Case 2: mhac_k (with predictor) --------------------------------
    print("\n[2/2] mhac_k (with predictor)")
    env_mhac = make_env(env_name, test_seeds)
    predictor = MHACPredictor(
        latent_dim=LATENT_DIM,
        num_actions=num_actions,
        num_layers=1,
        num_heads=2,
    )
    model_mhac = MHACTrainer(
        policy="MlpPolicy",
        env=env_mhac,
        predictor=predictor,
        lambda_pred=0.1,
        lambda_cons=0.1,
        policy_kwargs={
            "features_extractor_kwargs": {"latent_dim": LATENT_DIM},
            "net_arch": [],
        },
        n_steps=32,
        batch_size=32,
        verbose=0,
    )
    eval_env_mhac = make_env(env_name, test_seeds)
    data_mhac = collect_latent_snapshot(
        model=model_mhac,
        env=eval_env_mhac,
        n_episodes=N_EPISODES,
        horizons=HORIZONS,
    )
    _assert(data_mhac is not None, "collect returned None for mhac_k")
    _check_snapshot_dict(data_mhac, has_predictor=True, horizons=HORIZONS)
    env_mhac.close()
    eval_env_mhac.close()

    print("\nAll generate-mode checks passed.")


# ---------------------------------------------------------------------------
# Mode 2: verify an existing snapshot directory
# ---------------------------------------------------------------------------

def _run_verify_mode(snapshot_dir: Path) -> None:
    import numpy as np

    npz_files = sorted(snapshot_dir.glob("*.npz"))
    _assert(len(npz_files) > 0, f"No .npz files found in {snapshot_dir}")

    print(f"\nFound {len(npz_files)} snapshot(s) in {snapshot_dir}")
    for path in npz_files:
        print(f"\n  {path.name}")
        data = dict(np.load(str(path)))
        has_predictor = any(k.startswith("z_hat_") for k in data)
        _check_snapshot_dict(data, has_predictor=has_predictor, horizons=HORIZONS)

    print(f"\nAll {len(npz_files)} snapshot(s) verified.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Smoketest for latent snapshot collection.")
    p.add_argument(
        "--snapshot-dir",
        default=None,
        help="Verify .npz files in this directory instead of generating a fresh snapshot.",
    )
    args = p.parse_args()

    if args.snapshot_dir is not None:
        _run_verify_mode(Path(args.snapshot_dir))
    else:
        _run_generate_mode()


if __name__ == "__main__":
    main()
