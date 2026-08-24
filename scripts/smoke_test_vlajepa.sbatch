#!/usr/bin/env bash
#SBATCH --job-name=so101_smoke_vlajepa
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Sized for RWTH CLAIX-23 `c23g` (1/4 of a 4× H100 node).

# VLA-JEPA-only smoke test: 10 optimizer steps, batch 2. Confirms the pretrained
# checkpoint loads, the 6-DoF reinit_modules path works, PEFT wraps this custom policy,
# AV1 video decodes, and one fwd/bwd/step runs on a single H100.
# The full job (slurm/train_vla_jepa.sbatch) must not be launched until this exits 0.

set -uo pipefail

# Video decoding: force the `pyav` backend (see --dataset.video_backend below). The datasets are
# AV1, and lerobot's default torchcodec backend needs a system FFmpeg with a *software* AV1
# decoder — the RWTH FFmpeg modules ship only CUVID (hardware) AV1 and fail on CPU with
# "Could not push packet to decoder: Function not implemented". PyAV bundles its own
# dav1d-enabled FFmpeg, so it decodes AV1 with no module/LD_LIBRARY_PATH juggling.

source "$HOME/lerobot/.venv/bin/activate"
: "${SCRATCH:=${HPCWORK:?set SCRATCH or HPCWORK to a fast, large filesystem}}"
export HF_HOME=${HF_HOME:-$SCRATCH/hf}
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p logs "$SCRATCH/train/smoke"
rm -rf "$SCRATCH/train/smoke/vlajepa"

# `shubham4413/so101_wm` has NO `v3.0` git tag on the Hub (`so101_vla` does), so lerobot's
# get_safe_version() revision lookup raises before the dataset ever loads. Pointing
# --dataset.root at the local snapshot skips that lookup entirely.
# Permanent fix: HfApi().create_tag("shubham4413/so101_wm", tag="v3.0", repo_type="dataset").
DATA_ROOT=${DATA_ROOT:-$(ls -d "$HF_HOME"/hub/datasets--shubham4413--so101_wm/snapshots/*/ | head -1)}
echo "DATA_ROOT=$DATA_ROOT"

# The pretrained VLA-JEPA config names its two cameras exterior_1_left / exterior_2_left;
# our dataset uses left / fpv. Map our keys onto the config's slots. The slot assignment is
# arbitrary (both are plain camera slots; the input projections are reinit'd anyway) but MUST
# be identical for training, offline eval, and deployment — keep this map in sync everywhere.
RENAME_MAP='{"observation.images.left":"observation.images.exterior_1_left","observation.images.fpv":"observation.images.exterior_2_left"}'

nvidia-smi
echo "=================== VLA-JEPA smoke (10 steps, LoRA + WM) ==================="
# Watch for PEFT's "trainable params: ... || all params: ..." line and confirm the
# trainable set includes Qwen LoRA adapters + model.action_model + model.video_predictor.
lerobot-train \
  --policy.path=lerobot/VLA-JEPA-Pretrain \
  --dataset.repo_id=shubham4413/so101_wm \
  --dataset.root="$DATA_ROOT" \
  --dataset.video_backend=pyav \
  --tolerance_s=1e-3 \
  --policy.device=cuda \
  --policy.gripper_dim=5 \
  --policy.reinit_modules='["model.action_model.action_encoder","model.action_model.action_decoder","model.action_model.state_encoder"]' \
  --peft.method_type=LORA --peft.r=16 --peft.lora_alpha=32 \
  --peft.target_modules='.*model\.qwen\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)' \
  --peft.full_training_modules='["model.action_model","model.video_predictor"]' \
  --output_dir="$SCRATCH/train/smoke/vlajepa" \
  --policy.push_to_hub=false \
  --rename_map="$RENAME_MAP" \
  --batch_size=2 --steps=10 --save_freq=10 \
  --log_freq=1 --num_workers=4 --wandb.enable=false
VJ_RC=$?

echo "=================== SUMMARY ==================="
echo "vla_jepa smoke exit code: $VJ_RC   (0 = ok)"
echo "If it failed on the PEFT wrap, see README 'Fallback: full fine-tune'."
exit $VJ_RC
