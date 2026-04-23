"""
Main training entry point.

Usage:
    python scripts/train.py --condition baseline --env fourrooms
    python scripts/train.py --condition mhac_k --env fourrooms
    python scripts/train.py --condition baseline --env multiroom
"""

import argparse
import os
import sys
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback

# Resolve the project root regardless of CWD or how __file__ is set
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env.wrappers import get_seed_split
from src.env import wrappers as minigrid_factory
from src.env import miniworld_wrappers as miniworld_factory
from src.training.ppo_trainer import MHACTrainer
from src.utils.logging import WandBEvalCallback, LatentSnapshotCallback

ENV_NAMES = {
    "fourrooms":           "MiniGrid-FourRooms-v0",
    "multiroom":           "MiniGrid-MultiRoom-N6-v0",
    "multiroom_n4":        "MiniGrid-MultiRoom-N4-S5-v0",
    "multiroom_n2":        "MiniGrid-MultiRoom-N2-S4-v0",
    "doorkey":             "MiniGrid-DoorKey-8x8-v0",
    "miniworld_hallway":   "MiniWorld-Hallway-v0",
    "miniworld_oneroom":   "MiniWorld-OneRoom-v0",
    "miniworld_fourrooms": "MiniWorld-FourRooms-v0",
}


def _is_miniworld(env_key: str) -> bool:
    return env_key.startswith("miniworld_")


def _factory_for(env_key: str):
    """Return the env-factory module matching the selected env family."""
    return miniworld_factory if _is_miniworld(env_key) else minigrid_factory

# Hyperparameters fixed across all conditions (from base.yaml / plan)
LATENT_DIM    = 256
N_STEPS       = 128   # rollout length per env — must be >= K
N_ENVS        = 16
TOTAL_STEPS   = 5_000_000
LEARNING_RATE = 3e-4
BATCH_SIZE    = 256
N_EPOCHS      = 4
HORIZON       = 5     # K

# Aux loss weights per condition
CONDITION_CFG = {
    "baseline":          dict(lambda_pred=0.0, lambda_cons=0.0),
    "one_step":          dict(lambda_pred=0.1, lambda_cons=0.0, horizon_override=1),
    "k_step_no_cons":    dict(lambda_pred=0.1, lambda_cons=0.0),
    "mhac_k":            dict(lambda_pred=0.1, lambda_cons=0.05),
    "k_step_double_pred":dict(lambda_pred=0.2, lambda_cons=0.0),
    "no_action":         dict(lambda_pred=0.1, lambda_cons=0.0, no_action=True),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=list(CONDITION_CFG.keys()))
    p.add_argument("--env", required=True, choices=list(ENV_NAMES.keys()))
    p.add_argument("--seed-pool", type=int, default=500)
    p.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100_000,
        help="Save an intermediate checkpoint every N environment timesteps",
    )
    p.add_argument(
        "--latent-snapshot-dir",
        default="latent_snapshots",
        help="Directory for latent snapshot NPZs (one subdir per run). "
             "Pass empty string to disable.",
    )
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--seed", type=int, default=42, help="Global random seed")
    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--eval-freq", type=int, default=50_000,
                   help="Evaluate every N timesteps (W&B only)")
    p.add_argument("--eval-episodes", type=int, default=50,
                   help="Episodes per evaluation")
    p.add_argument("--lambda-cons", type=float, default=None,
                   help="Override λ_cons from CONDITION_CFG (e.g. 0.05)")
    p.add_argument("--lambda-recon", type=float, default=0.0,
                   help="Weight for grid reconstruction auxiliary loss (default 0 = disabled)")
    p.add_argument("--n-steps", type=int, default=N_STEPS,
                   help="PPO rollout steps per env (default 128; use 512-1024 for MultiRoom)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CONDITION_CFG[args.condition]
    env_name = ENV_NAMES[args.env]
    horizon = cfg.get("horizon_override", HORIZON)
    lambda_pred = cfg["lambda_pred"]
    lambda_cons = args.lambda_cons if args.lambda_cons is not None else cfg["lambda_cons"]
    use_action = not cfg.get("no_action", False)

    lambda_recon = args.lambda_recon

    run_name = f"{args.env}_{args.condition}_seed{args.seed}"
    ckpt_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # --- Seed split (frozen for all experiments) ---
    train_seeds, test_seeds = get_seed_split(
        seed_pool=args.seed_pool, test_fraction=0.2, rng_seed=0
    )
    print(f"Train seeds: {len(train_seeds)}  |  Test seeds: {len(test_seeds)}")

    # --- Environments ---
    factory = _factory_for(args.env)
    vec_envs = factory.make_vec_envs(env_name, train_seeds, num_envs=N_ENVS)

    # --- Predictor (None for baseline) ---
    predictor = None
    if lambda_pred > 0.0 or lambda_cons > 0.0:
        from src.models.predictor import MHACPredictor
        # Use a wrapped sample env so NavActions-style action restrictions are reflected.
        sample_env = factory.make_env(env_name, train_seeds)
        num_actions = sample_env.action_space.n
        sample_env.close()
        predictor = MHACPredictor(
            latent_dim=LATENT_DIM,
            num_actions=num_actions,
            num_layers=2,
            num_heads=4,
            use_action_conditioning=use_action,
        )

    # --- Decoder (optional) ---
    decoder = None
    if lambda_recon > 0.0:
        from src.models.decoder import GridDecoder
        decoder = GridDecoder(latent_dim=LATENT_DIM)

    # --- Trainer ---
    model = MHACTrainer(
        policy="MlpPolicy",
        env=vec_envs,
        predictor=predictor,
        lambda_pred=lambda_pred,
        lambda_cons=lambda_cons,
        horizon=horizon,
        decoder=decoder,
        lambda_recon=lambda_recon,
        n_steps=args.n_steps,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=args.log_dir if args.tensorboard else None,
        seed=args.seed,
    )

    print(f"\nStarting run: {run_name}")
    print(f"  lambda_pred={lambda_pred}  lambda_cons={lambda_cons}  K={horizon}  lambda_recon={lambda_recon}\n")

    callbacks = []

    if args.wandb:
        train_eval_env = factory.make_env(env_name, train_seeds)
        test_eval_env  = factory.make_env(env_name, test_seeds)
        callbacks.append(
            WandBEvalCallback(
                train_env=train_eval_env,
                test_env=test_eval_env,
                n_eval_episodes=args.eval_episodes,
                eval_freq=args.eval_freq,
                run_name=run_name,
                condition=args.condition,
                env_name=args.env,
                verbose=1,
            )
        )

    if args.checkpoint_freq > 0:
        # SB3's callback frequency is counted in env.step() calls, not raw
        # timesteps, so divide by the number of parallel envs.
        save_freq = max(args.checkpoint_freq // N_ENVS, 1)
        callbacks.append(
            CheckpointCallback(
                save_freq=save_freq,
                save_path=ckpt_dir,
                name_prefix=run_name,
            )
        )

        if args.latent_snapshot_dir:
            snapshot_env = factory.make_env(env_name, test_seeds)
            callbacks.append(
                LatentSnapshotCallback(
                    eval_env=snapshot_env,
                    snapshot_freq=save_freq,   # same cadence as checkpoints
                    snapshot_dir=args.latent_snapshot_dir,
                    run_name=run_name,
                    n_episodes=5,
                    verbose=1,
                )
            )

    model.learn(
        total_timesteps=args.total_steps,
        tb_log_name=run_name,
        reset_num_timesteps=True,
        callback=callbacks or None,
    )

    final_path = os.path.join(ckpt_dir, "final.zip")
    model.save(final_path)
    print(f"\nSaved to {final_path}")

    if decoder is not None:
        import torch
        decoder_path = os.path.join(ckpt_dir, "decoder.pt")
        torch.save(decoder.state_dict(), decoder_path)
        print(f"Saved decoder to {decoder_path}")


if __name__ == "__main__":
    main()
