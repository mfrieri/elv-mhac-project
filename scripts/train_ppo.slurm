#!/bin/bash
#SBATCH --job-name=mhac-train
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mxf5227@nyu.edu


set -euo pipefail

CONDITION="${1:-baseline}"
ENV_NAME="${2:-fourrooms}"
TOTAL_STEPS="${3:-5000000}"
CHECKPOINT_FREQ="${4:-100000}"

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

singularity exec --nv \
  --overlay "${OVERLAY_PATH}:ro" \
  "$SIF_PATH" \
  /bin/bash -lc "
    set -euo pipefail
    cd '$PROJECT_ROOT'
    source /ext3/env.sh
    conda activate mhac
    python scripts/train.py \
      --condition '$CONDITION' \
      --env '$ENV_NAME' \
      --total-steps '$TOTAL_STEPS' \
      --checkpoint-freq '$CHECKPOINT_FREQ' \
      --tensorboard \
      --wandb
  "

echo "Finished at $(date)"
