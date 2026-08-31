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

HPC_ROOT=${HPC_ROOT:-"${HPCWORK:-/hpcwork/$USER}"}
BASE="$HPC_ROOT/so101"
PROJECT="$HPC_ROOT/so101_physical_ai"
: "${RUN:?set RUN to the diffusion training directory to resume}"
CONFIG="$RUN/checkpoints/070000/pretrained_model/train_config.json"

export PYTHONNOUSERSITE=1
export HF_HOME="$BASE/hf_cache"
export HF_LEROBOT_HOME="$BASE/hf_cache/lerobot"
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
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
