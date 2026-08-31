#!/usr/bin/env bash
#SBATCH --job-name=so101_vlajepa
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=23:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# RWTH CLAIX-23 configuration; adjust partition and GRES at other sites.

set -euo pipefail

# PyAV supplies a software AV1 decoder independent of the cluster FFmpeg build.

source "$HOME/lerobot/.venv/bin/activate"
# CLAIX exposes fast parallel storage as $HPCWORK; other sites use $SCRATCH.
: "${SCRATCH:=${HPCWORK:?set SCRATCH or HPCWORK to a fast, large filesystem}}"
export HF_HOME=${HF_HOME:-$SCRATCH/hf}
export HF_HUB_ENABLE_HF_TRANSFER=1
export OUT=${OUT:-$SCRATCH/train/so101_wm_vlajepa}
# LeRobot creates the output directory and rejects an existing path.
mkdir -p logs

# Loading from a concrete snapshot avoids dependency on Hub revision tags.
DATA_ROOT=${DATA_ROOT:-$(ls -d "$HF_HOME"/hub/datasets--shubham4413--so101_wm/snapshots/*/ | head -1)}
echo "DATA_ROOT=$DATA_ROOT"

# Map dataset camera names to the pretrained policy input slots.
RENAME_MAP='{"observation.images.left":"observation.images.exterior_1_left","observation.images.fpv":"observation.images.exterior_2_left"}'

# A 1 ms tolerance accommodates PyAV timestamp precision while remaining well
# below the 33 ms camera period. LoRA adapts Qwen while the action and world
# model heads remain fully trainable. Episode selection is computed inside the
# job because Slurm treats commas in JSON lists as export separators.
EP_ARG=()
if [ "${TRAIN_SPLIT:-1}" = "1" ]; then
  EPISODES=$(python data/make_splits.py --repo-id shubham4413/so101_wm --root "$DATA_ROOT" --print train)
  echo "train-split episodes: ${#EPISODES} chars, starts ${EPISODES:0:40}"
  EP_ARG=(--dataset.episodes="$EPISODES")
fi

lerobot-train \
  --policy.path=lerobot/VLA-JEPA-Pretrain \
  --policy.repo_id=shubham4413/so101_wm_vlajepa \
  --policy.push_to_hub=false \
  --rename_map="$RENAME_MAP" \
  --dataset.repo_id=shubham4413/so101_wm \
  --dataset.root="$DATA_ROOT" \
  --dataset.video_backend=pyav \
  --tolerance_s=1e-3 \
  --policy.device=cuda \
  --policy.gripper_dim=5 \
  --policy.reinit_modules='["model.action_model.action_encoder","model.action_model.action_decoder","model.action_model.state_encoder"]' \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=32 \
  --peft.target_modules='.*model\.qwen\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)' \
  --peft.full_training_modules='["model.action_model","model.video_predictor"]' \
  --output_dir="$OUT" \
  --batch_size=8 \
  --steps=30000 \
  --save_freq=10000 \
  --log_freq=100 \
  --num_workers=8 \
  --wandb.enable=${WANDB:-false} \
  "${EP_ARG[@]}"

echo "Done. Checkpoints in $OUT"
