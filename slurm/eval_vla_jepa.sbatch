#!/usr/bin/env bash
#SBATCH --job-name=so101_vj_eval
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

source /home/dl125352/lerobot/.venv/bin/activate

export HF_HOME=${HF_HOME:-/hpcwork/dl125352/hf}
export HF_XET_HIGH_PERFORMANCE=1

PROJECT_DIR=${PROJECT_DIR:-/home/dl125352/so101_physical_ai}
POLICY_PATH=${POLICY_PATH:-/hpcwork/dl125352/train/so101_wm_vlajepa/checkpoints/030000/pretrained_model}
DATASET_ROOT=${DATASET_ROOT:-/hpcwork/dl125352/hf/hub/datasets--shubham4413--so101_wm/snapshots/96e2ecc061a02d2952083128350f6815de30cb9f}
MAX_EPISODES=${MAX_EPISODES:-0}
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/eval/out/vlajepa_030000}

cd "$PROJECT_DIR"
mkdir -p logs "$OUT_DIR"

python eval/offline_action_error.py \
  --policy-path "$POLICY_PATH" \
  --repo-id shubham4413/so101_wm \
  --dataset-root "$DATASET_ROOT" \
  --out-dir "$OUT_DIR" \
  --device cuda \
  --video-backend pyav \
  --tolerance-s 1e-3 \
  --max-episodes "$MAX_EPISODES" \
  --rename-map '{"observation.images.left":"observation.images.exterior_1_left","observation.images.fpv":"observation.images.exterior_2_left"}'
