# MHAC — Multi-Horizon Action Consistency with Temporal Consistency

A research RL project that extends **PPO** with two auxiliary losses for representation learning on partially observable navigation tasks:

- **L_pred** — *k-step latent prediction*: a Transformer predictor learns to forecast the encoder's future latents.
- **L_cons** — *temporal consistency*: chained rollout predictions are pulled toward direct k-step predictions.

The project currently trains on MiniGrid navigation environments and ablates which parts of the auxiliary objective actually shape the encoder's latent geometry, sample efficiency, and generalization to held-out seeds.

> Course: NYU DS-GA 3001 (Embodied Learning & Vision) · ELV Project · 2026 Spring

---

## Table of contents

- [MHAC — Multi-Horizon Auxiliary Consistency](#mhac--multi-horizon-auxiliary-consistency)
  - [Table of contents](#table-of-contents)
  - [Motivation](#motivation)
  - [Method](#method)
    - [Data flow](#data-flow)
    - [Stop-gradient convention (important)](#stop-gradient-convention-important)
    - [Key modules](#key-modules)
  - [Repository layout](#repository-layout)
  - [Installation](#installation)
  - [Quick start](#quick-start)
    - [Train locally](#train-locally)
    - [Common flags](#common-flags)
  - [Training conditions](#training-conditions)
  - [Configuration](#configuration)
  - [HPC / SLURM](#hpc--slurm)
  - [Evaluation, probes, and analysis](#evaluation-probes-and-analysis)
    - [Save latents from a trained checkpoint](#save-latents-from-a-trained-checkpoint)
    - [Inspect / probe latents](#inspect--probe-latents)
    - [Optional: decoder diagnostic](#optional-decoder-diagnostic)
  - [Logging](#logging)
    - [W\&B](#wb)
    - [TensorBoard](#tensorboard)
  - [Findings](#findings)
  - [Reproducibility](#reproducibility)
  - [Citation](#citation)

---

## Motivation

Vanilla PPO learns a value-and-policy head on top of CNN features without any constraint on what the latent space *means*. On small MiniGrid navigation tasks this works, but the encoder is free to overfit to seed-specific surface features and the representation has no inductive bias toward task geometry (e.g., distance-to-goal).

MHAC asks: **does forcing the encoder to support multi-step future prediction in latent space — without ever decoding to pixels — produce a more task-relevant representation, better sample efficiency, and better generalization to unseen seeds?**

Each ablation in this repo (`one_step`, `k_step_no_cons`, `mhac_k`, `k_step_double_pred`) is designed to isolate a single mechanism in that objective so the answer is interpretable, not just measurable.

---

## Method

### Data flow

```
MiniGrid obs (H, W, C)
  → ImageToCHW wrapper → (C, H, W) float32 in [0, 1]
  → CNNEncoder         → z ∈ R^256
  → PolicyHead         → π(a | z), V(z)
  → PPO update          (standard)
  → Aux loss pass       (only if λ_pred > 0 or λ_cons > 0)
       Transformer Predictor:
         forward_all_horizons(z_t, a_{t..t+K-1})  → z_hat_direct  ∈ R^(B, K, 256)
         chain_with_grad(z_t, a_{t..t+K-1})       → z_hat_chain   ∈ R^(B, K, 256)
       L_pred = cosine_loss(z_hat_direct_k, sg(z_target_k))
       L_cons = cosine_loss(z_hat_chain_k,  sg(z_hat_direct_k))
       total  = λ_pred · L_pred + λ_cons · L_cons
```

### Stop-gradient convention (important)

| Loss | Where the stop-grad sits | Gradients flow into |
| --- | --- | --- |
| **L_pred** | the *target* `z_{t+k}` (encoder pass on future obs) | encoder + predictor |
| **L_cons** | the *direct* prediction (acts as anchor) | predictor (chained rollout only) |

This is what isolates "predict the future" from "be self-consistent under chained rollouts." Get this wrong and the two losses collapse onto each other.

### Key modules

| Module | Purpose |
| --- | --- |
| [src/models/encoder.py](src/models/encoder.py) | `CNNEncoder` for MiniGrid `(3, 7, 7)` → 256-d latent |
| [src/models/predictor.py](src/models/predictor.py) | Transformer predictor: direct k-step + autoregressive chain |
| [src/models/policy.py](src/models/policy.py) | `MHACFeaturesExtractor` and actor / critic heads |
| [src/models/decoder.py](src/models/decoder.py) | Optional `GridDecoder` (diagnostic only — see Finding 3) |
| [src/training/ppo_trainer.py](src/training/ppo_trainer.py) | `MHACTrainer` — SB3 PPO + second aux backward pass |
| [src/training/losses.py](src/training/losses.py) | `prediction_loss`, `consistency_loss`, `reconstruction_loss` |
| [src/env/wrappers.py](src/env/wrappers.py) | MiniGrid wrappers + frozen 80/20 seed split |
| [src/env/miniworld_wrappers.py](src/env/miniworld_wrappers.py) | MiniWorld 3D variants |
| [src/evaluation/eval.py](src/evaluation/eval.py) | Success rate / reward eval |
| [src/evaluation/diagnostics.py](src/evaluation/diagnostics.py) | `drift_curve` and other latent diagnostics |
| [src/utils/logging.py](src/utils/logging.py) | `WandBEvalCallback`, `LatentSnapshotCallback` |

---

## Repository layout

```
elv-mhac-project/
├── configs/                YAML configs (base + per-env overrides)
│   ├── base.yaml
│   ├── fourrooms/
│   ├── multiroom/
│   └── miniworld/
├── scripts/                Entry points
│   ├── train.py            Main training entry point
│   ├── evaluate.py
│   ├── save_latents.py     Snapshot latents from a checkpoint
│   ├── analyze_latents.py  Linear probes + effective rank
│   ├── inspect_latents.py
│   ├── decode_latents.py   Diagnostic decoder (Finding 3)
│   ├── smoketest_latent_snapshots.py
│   ├── train_ppo.slurm     SLURM submission for MiniGrid
│   └── train_miniworld.slurm
├── src/                    Library code (see table above)
├── checkpoints/            Saved SB3 zips (one subdir per run)
├── latents/                Latent snapshots
├── logs/                   TensorBoard logs
├── results/
│   ├── findings.md         Presentation-ready results log
│   ├── probes.md           Probe-table results
│   └── figures/
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
```

---

## Installation

Tested on Python 3.9+ with CUDA 12.x (PyTorch 2.8). CPU-only works for the small MiniGrid models, but training a full 5M-step run is much slower.

```bash
# 1. Create and activate an environment
python -m venv .venv && source .venv/bin/activate
# (or: conda create -n mhac python=3.10 && conda activate mhac)

# 2. Install runtime deps
pip install -r requirements.txt

# 3. Optional dev extras
pip install -r requirements-dev.txt

# 4. Install this package in editable mode
pip install -e .
```

Headline runtime deps: `torch==2.8.0`, `stable_baselines3==2.7.1`, `gymnasium==1.1.1`, `minigrid==3.0.0`, `wandb==0.25.1`.

> **Note on MiniWorld.** MiniWorld pulls in `pyglet` and needs an X display just to register its envs, so it is *lazy-imported* in [scripts/train.py](scripts/train.py#L42) and only loaded when an `--env miniworld_*` is selected.

---

## Quick start

### Train locally

```bash
# Vanilla PPO baseline
python scripts/train.py --condition baseline --env fourrooms

# Full MHAC (k-step prediction + chained consistency)
python scripts/train.py --condition mhac_k --env fourrooms --wandb --tensorboard

# Larger rollout for MultiRoom (recommended)
python scripts/train.py --condition mhac_k --env multiroom --n-steps 512

# Override λ_cons or K from the command line
python scripts/train.py --condition mhac_k --env fourrooms --lambda-cons 0.05
python scripts/train.py --condition mhac_k --env fourrooms --k 10
```

### Common flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--condition` | required | one of [`baseline`, `one_step`, `k_step_no_cons`, `mhac_k`, `k_step_double_pred`] |
| `--env` | required | one of [`fourrooms`, `multiroom`, `multiroom_n4`, `multiroom_n2`, `doorkey`, `unlock_pickup`, `miniworld_hallway`, `miniworld_oneroom`, `miniworld_fourrooms`] |
| `--total-steps` | `5_000_000` | total environment timesteps |
| `--seed` | `42` | global RNG seed |
| `--n-steps` | `128` | PPO rollout length per env (must be ≥ K; use 512–1024 on MultiRoom) |
| `--k` | condition default | override prediction horizon K (used for K ablations) |
| `--lambda-cons` | condition default | override λ_cons |
| `--lambda-recon` | `0.0` | weight on grid-reconstruction loss (decoder turned on if > 0) |
| `--checkpoint-freq` | `100_000` | save an intermediate checkpoint every N timesteps |
| `--wandb` | off | enable W&B logging |
| `--tensorboard` | off | enable TensorBoard logging |

---

## Training conditions

Six conditions are wired into [scripts/train.py](scripts/train.py#L63). Each isolates one piece of the auxiliary objective.

| Condition | λ_pred | λ_cons | K | Action-cond? | What it tests |
| --- | --- | --- | --- | --- | --- |
| `baseline` | 0.0 | 0.0 | — | — | Pure PPO. Reference. |
| `one_step` | 0.1 | 0.0 | 1 | yes | Single-horizon prediction only. |
| `k_step_no_cons` | 0.1 | 0.0 | 5 | yes | K-step direct prediction only — no chained rollout. |
| **`mhac_k`** | **0.1** | **0.05** | **5** | **yes** | **Full method: direct prediction + chained consistency.** |
| `k_step_double_pred` | 0.2 | 0.0 | 5 | yes | Doubles L_pred weight — confounder check for L_cons. |
| `no_action` | 0.1 | 0.0 | 5 | **no** | Predictor with action input removed. |

---

## Configuration

Hyperparameters live in [configs/](configs/). [configs/base.yaml](configs/base.yaml) defines the cross-condition defaults; per-env directories override only what differs.

Key defaults (also fixed in [scripts/train.py](scripts/train.py#L52)):

| Parameter | Value |
| --- | --- |
| `LATENT_DIM` | 256 |
| `K` (prediction horizon) | 5 |
| `N_ENVS` | 16 (script) / 4 (base.yaml) |
| `N_STEPS` (PPO rollout) | 128 |
| `LR` | 3e-4 |
| `BATCH_SIZE` | 256 |
| `N_EPOCHS` | 4 |
| `TOTAL_STEPS` | 5,000,000 |
| `λ_pred / λ_cons` | 0.1 / 0.05 (mhac_k) |

> The CLI flags in [scripts/train.py](scripts/train.py) are the **authoritative** source of training hyperparameters. The YAML files are kept as a human-readable record of the intended sweep.

---

## HPC / SLURM

This project is set up for the NYU HPC cluster. Submit a single run with:

```bash
sbatch scripts/train_ppo.slurm <condition> <env> <total_steps> <checkpoint_freq> <seed> [lambda_cons] [lambda_recon] [k]

# Example: full MHAC, FourRooms, 5M steps, save every 100K, seed 42
sbatch scripts/train_ppo.slurm mhac_k fourrooms 5000000 100000 42

# Example: K ablation
sbatch scripts/train_ppo.slurm mhac_k fourrooms 5000000 100000 42 "" "" 10
```

The submit script expects:

- a Singularity overlay at `$OVERLAY_PATH` (defaults to `/scratch/$USER/my_env.ext3`),
- the cluster's CUDA 12.1 image at `$SIF_PATH`,
- a conda env named `$CONDA_ENV_NAME` (default `mhac`).

W&B and TensorBoard are both enabled by default in the SLURM wrapper. See [scripts/train_ppo.slurm](scripts/train_ppo.slurm) for the full singularity / conda invocation. A separate [scripts/train_miniworld.slurm](scripts/train_miniworld.slurm) handles the 3D MiniWorld runs.

---

## Evaluation, probes, and analysis

### Save latents from a trained checkpoint

```bash
python scripts/save_latents.py --checkpoint checkpoints/<run>/final.zip
```

This rolls the trained policy on held-out **test seeds** and writes per-step `(z, action, reward, info)` snapshots under `latents/<run>/`.

### Inspect / probe latents

```bash
# Sanity check that snapshots are well-formed
python scripts/smoketest_latent_snapshots.py

# Linear probes (Ridge / LogReg, 5-fold CV) and effective rank
python scripts/analyze_latents.py
python scripts/inspect_latents.py --snapshots latents/<run>
```

The probe pipeline writes results into [results/probes.md](results/probes.md). Quantities measured per condition:

- **`R²(goal_dist)`** — linear decodability of distance-to-goal from frozen latents (the headline probe; see Finding 2).
- **`R²(agent_x)`, `R²(agent_y)`** — sanity checks (negative under partial observability, by design).
- **`acc(direction)`** — chance is 0.25.
- **`eff_rank`** — effective rank of latent activations (collapse check).

### Optional: decoder diagnostic

```bash
# Train MHAC with the diagnostic decoder turned on
python scripts/train.py --condition mhac_k --env fourrooms --lambda-recon 0.1

# Visualize what predicted latents decode to
python scripts/decode_latents.py
```

Important: the decoder is a *diagnostic*, not part of MHAC. See Finding 3 for what it revealed and what it does not establish.

---

## Logging

### W&B

Enable with `--wandb`. Project name `mhac` (override in [configs/base.yaml](configs/base.yaml)). Headline panels:

| Metric | Meaning |
| --- | --- |
| `eval/train_sr`, `eval/test_sr` | Success rate on train- and test-seed splits |
| `eval/gen_gap` | `train_sr − test_sr` (generalization gap) |
| `aux/loss_pred_k{k}` | Per-horizon prediction loss |
| `aux/loss_cons` | Consistency loss |
| `aux/drift_k{k}` | Cosine distance between direct and chained predictions at horizon `k` |

The `WandBEvalCallback` evaluates every `--eval-freq` env steps over `--eval-episodes` episodes (defaults: 50K / 50). The `LatentSnapshotCallback` writes per-checkpoint latent NPZs to `--latent-snapshot-dir`.

### TensorBoard

```bash
python scripts/train.py --condition mhac_k --env fourrooms --tensorboard
tensorboard --logdir=logs
```

---

## Findings

Final, presentation-ready results live in [results/findings.md](results/findings.md). Every entry leads with the claim, then the numbers, then the caveats. Brief headlines:

| # | Headline |
| --- | --- |
| 1 | Drift is **not** what L_cons controls — L_pred alone drives drift to zero. |
| 2 | Prediction aux losses (any form) regularize task-relevant geometry; chained rollout is **not** specifically responsible. |
| 3 | The decoder is a *diagnostic*, not a product — predicted latents decode to neutral grids despite high cosine similarity. |
| 4 | The chained rollout in L_cons provides **no measurable benefit** over scaling L_pred. |
| 5 | K=5 is the sweet spot; longer horizons regularize but with diminishing returns. |
| 6 | Every prediction-based aux loss beats baseline; **action conditioning is unnecessary** and slightly destabilizing. |
| 7 | K=5 prediction losses produce a ~40% sample-efficiency speedup; K=1 does not. |
| 8 | L_cons converges to ~0 but does not improve final long-horizon prediction quality. |

The simplest distillation:

> **Multi-horizon latent prediction is what does the work.** The chained-consistency rollout, the action conditioning, and the decoder all turned out to not be load-bearing on FourRooms.

---

## Reproducibility

- **Seed split.** The 80/20 train/test seed split (out of 500 candidate env seeds) is frozen and deterministic via `rng_seed=0` in [src/env/wrappers.py](src/env/wrappers.py). **Never alter this split mid-experiment** — every `findings.md` entry assumes it.
- **Five seeds per condition.** Headline numbers in [results/findings.md](results/findings.md) are reported as mean ± std over seeds `{42, 123, 456, 789, 1000}` at the 5M-step final checkpoint, 50 eval episodes per seed on held-out test seeds.
- **W&B as authoritative SR.** When the post-hoc `save_latents` SR and the in-training `WandBEvalCallback` SR disagree, the W&B value at the canonical step is the reported number (see Finding 4 for an example).

---

## Citation

If you use this code, please cite:

```bibtex
@misc{frieri2026mhac,
  author = {Michael Frieri and Chinmayee Gade},
  title  = {Multi-Horizon Action-Conditioned Prediction with Temporal Consistency},
  year   = {2026},
  note   = {NYU DS-GA 3001 ELV course project},
  url    = {https://github.com/mfrieri/elv-mhac-project}
}
```
