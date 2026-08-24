#!/usr/bin/env bash
#SBATCH --job-name=so101_dp_resume
#SBATCH --partition=c23g
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

BASE=/hpcwork/ay713634/so101
PROJECT=/hpcwork/ay713634/so101_physical_ai
RUN=/hpcwork/ay713634/train/so101_vla_diffusion_3080015
CONFIG="$RUN/checkpoints/070000/pretrained_model/train_config.json"

export PYTHONNOUSERSITE=1
export HF_HOME="$BASE/hf_cache"
export HF_LEROBOT_HOME="$BASE/hf_cache/lerobot"
export TORCH_HOME=/home/ay713634/.cache/torch
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT"
mkdir -p logs

test -f "$CONFIG"

echo "job=$SLURM_JOB_ID node=$(hostname) resume_config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun --unbuffered "$BASE/env/bin/lerobot-train" \
  --config_path="$CONFIG" \
  --resume=true \
  --save_freq=5000

echo "RESUME_OK output=$RUN final_step=100000"
