#!/usr/bin/env bash
#SBATCH --job-name=so101_diffusion
#SBATCH --partition=c23g
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=09:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

BASE=/hpcwork/ay713634/so101
PROJECT=/hpcwork/ay713634/so101_physical_ai
OUT=${OUT:-/hpcwork/ay713634/train/so101_vla_diffusion_${SLURM_JOB_ID}}

export PYTHONNOUSERSITE=1
export HF_HOME="$BASE/hf_cache"
export HF_LEROBOT_HOME="$BASE/hf_cache/lerobot"
export TORCH_HOME=/home/ay713634/.cache/torch
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT"
mkdir -p logs "$(dirname "$OUT")"

echo "job=$SLURM_JOB_ID node=$(hostname) output=$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun --unbuffered "$BASE/env/bin/lerobot-train" \
  --dataset.repo_id=shubham4413/so101_vla \
  --dataset.eval_split=0.15 \
  --dataset.video_backend=pyav \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.resize_shape='[240,320]' \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --batch_size=64 \
  --steps=100000 \
  --save_freq=5000 \
  --eval_steps=5000 \
  --max_eval_samples=2048 \
  --log_freq=200 \
  --num_workers=8 \
  --wandb.enable=false

echo "TRAINING_OK output=$OUT"
