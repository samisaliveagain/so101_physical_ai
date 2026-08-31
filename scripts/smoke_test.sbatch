#!/usr/bin/env bash
#SBATCH --job-name=so101_smoke
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Add site-specific --partition / --account.

# Ten-step validation for dataset decoding, model loading, and optimization.

set -uo pipefail
source "$HOME/lerobot/.venv/bin/activate"
export HF_HOME=${HF_HOME:-$SCRATCH/hf}
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p logs "$SCRATCH/train/smoke"

echo "=================== DIFFUSION smoke (10 steps) ==================="
lerobot-train \
  --dataset.repo_id=shubham4413/so101_vla \
  --policy.type=diffusion --policy.device=cuda \
  --output_dir="$SCRATCH/train/smoke/diffusion" \
  --batch_size=2 --steps=10 --save_freq=10 --eval_freq=100000 \
  --log_freq=1 --num_workers=4 --wandb.enable=false
DIFF_RC=$?
echo "diffusion smoke exit code: $DIFF_RC"

echo "=================== VLA-JEPA smoke (10 steps, LoRA + WM) ==================="
# The log should list Qwen LoRA adapters, the action head, and video predictor
# as trainable parameters.
lerobot-train \
  --policy.path=lerobot/VLA-JEPA-Pretrain \
  --dataset.repo_id=shubham4413/so101_wm \
  --policy.device=cuda \
  --policy.gripper_dim=5 \
  --policy.reinit_modules='["model.action_model.action_encoder","model.action_model.action_decoder","model.action_model.state_encoder"]' \
  --peft.method_type=LORA --peft.r=16 --peft.lora_alpha=32 \
  --peft.target_modules='.*model\.qwen\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)' \
  --peft.full_training_modules='["model.action_model","model.video_predictor"]' \
  --output_dir="$SCRATCH/train/smoke/vlajepa" \
  --batch_size=2 --steps=10 --save_freq=10 --eval_freq=100000 \
  --log_freq=1 --num_workers=4 --wandb.enable=false
VJ_RC=$?
echo "vla_jepa smoke exit code: $VJ_RC"

echo "=================== SUMMARY ==================="
echo "diffusion: $DIFF_RC   vla_jepa: $VJ_RC   (0 = ok)"
echo "If vla_jepa failed on the PEFT wrap, see README 'Fallback: full fine-tune'."
