#!/usr/bin/env bash
#SBATCH --job-name=so101_dp_smoke
#SBATCH --partition=c23g
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

HPC_ROOT=${HPC_ROOT:-"${HPCWORK:-/hpcwork/$USER}"}
BASE="$HPC_ROOT/so101"
PROJECT="$HPC_ROOT/so101_physical_ai"
OUT=${OUT:-$HPC_ROOT/train/so101_vla_diffusion_smoke_${SLURM_JOB_ID}}

export PYTHONNOUSERSITE=1
export HF_HOME="$BASE/hf_cache"
export HF_LEROBOT_HOME="$BASE/hf_cache/lerobot"
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT"
mkdir -p logs "$(dirname "$OUT")"

echo "job=$SLURM_JOB_ID node=$(hostname) output=$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun --unbuffered "$BASE/env/bin/lerobot-train" \
  --dataset.repo_id=shubham4413/so101_vla \
  --dataset.eval_split=0.05 \
  --dataset.video_backend=pyav \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.resize_shape='[240,320]' \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --batch_size=64 \
  --steps=5 \
  --save_freq=5 \
  --eval_steps=5 \
  --max_eval_samples=64 \
  --log_freq=1 \
  --num_workers=4 \
  --wandb.enable=false

echo "SMOKE_OK output=$OUT"
