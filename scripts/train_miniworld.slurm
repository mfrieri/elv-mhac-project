#!/bin/bash
#SBATCH --job-name=mhac-miniworld
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=24G
#SBATCH --requeue
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mxf5227@nyu.edu


set -euo pipefail

CONDITION="${1:-baseline}"
ENV_NAME="${2:-miniworld_hallway}"
TOTAL_STEPS="${3:-3000000}"
CHECKPOINT_FREQ="${4:-100000}"
SEED="${5:-42}"
LAMBDA_CONS="${6:-}"
LAMBDA_RECON="${7:-}"

OVERLAY_PATH="${OVERLAY_PATH:-/scratch/$USER/my_env.ext3}"
SIF_PATH="${SIF_PATH:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mhac}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"

echo "Starting training at $(date)"
echo "Project root: $PROJECT_ROOT"
echo "Condition: $CONDITION"
echo "Environment: $ENV_NAME"
echo "Total steps: $TOTAL_STEPS"
echo "Checkpoint freq: $CHECKPOINT_FREQ"
echo "Overlay: $OVERLAY_PATH"
echo "Image: $SIF_PATH"
echo "Conda env: $CONDA_ENV_NAME"

cd "$PROJECT_ROOT"

# MiniWorld uses pyglet 1.5, which opens an X display at import time.
# xvfb-run provides a virtual display so rendering works on headless nodes.
singularity exec --nv \
  --overlay "${OVERLAY_PATH}:ro" \
  "$SIF_PATH" \
  /bin/bash -lc "
    set -euo pipefail
    cd '$PROJECT_ROOT'
    source /ext3/env.sh
    conda activate mhac
    xvfb-run -a python scripts/train.py \
      --condition '$CONDITION' \
      --env '$ENV_NAME' \
      --total-steps '$TOTAL_STEPS' \
      --checkpoint-freq '$CHECKPOINT_FREQ' \
      --seed '$SEED' \
      ${LAMBDA_CONS:+--lambda-cons '$LAMBDA_CONS'} \
      ${LAMBDA_RECON:+--lambda-recon '$LAMBDA_RECON'} \
      --tensorboard \
      --wandb
  "

echo "Finished at $(date)"
