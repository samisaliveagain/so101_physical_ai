#!/usr/bin/env bash
#SBATCH --job-name=so101_vlajepa
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=23:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Sized for RWTH CLAIX-23 `c23g` (4× H100-96GB, 96 cores, 500 GB per node -> 1/4 node).
# On a site without `c23g`, swap --partition/--gres for the local H100 equivalent.

set -euo pipefail

# ---- environment ----
# Video decoding: force the `pyav` backend (see --dataset.video_backend below). The datasets are
# AV1, and lerobot's default torchcodec backend needs a system FFmpeg with a *software* AV1
# decoder — the RWTH FFmpeg modules ship only CUVID (hardware) AV1 and fail on CPU with
# "Could not push packet to decoder: Function not implemented". PyAV bundles its own
# dav1d-enabled FFmpeg, so it decodes AV1 with no module/LD_LIBRARY_PATH juggling.

source "$HOME/lerobot/.venv/bin/activate"
# CLAIX exposes fast parallel storage as $HPCWORK; other sites use $SCRATCH.
: "${SCRATCH:=${HPCWORK:?set SCRATCH or HPCWORK to a fast, large filesystem}}"
export HF_HOME=${HF_HOME:-$SCRATCH/hf}
export HF_HUB_ENABLE_HF_TRANSFER=1
export OUT=${OUT:-$SCRATCH/train/so101_wm_vlajepa}
# Do NOT pre-create $OUT: lerobot's cfg.validate() aborts if output_dir already exists and
# --resume is false (anti-clobber guard). Let lerobot create it. Only make the logs dir.
mkdir -p logs

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

# --tolerance_s below is raised from the 1e-4 default: with the pyav backend some frame timestamps
# land exactly at 0.1ms tolerance and raise FrameTimestampError ("0.0001 > 0.0001"). 1e-3 absorbs
# pyav's timestamp precision jitter and stays << the 33ms/frame period at 30fps (no wrong frames).

# =====================================================================================
# VLA-JEPA fine-tune: LoRA on Qwen + world model ON.
#
# WHY LoRA (not freeze_qwen): configuration_vla_jepa.py forces `enable_world_model=False`
# whenever `freeze_qwen=true` (no grad reaches the world model). LoRA keeps gradients
# flowing through Qwen adapters, so the JEPA world-model objective stays meaningful while
# the 2B backbone's optimizer/grad memory stays tiny -> fits one H100.
#
# Layout of trainable params:
#   - LoRA adapters on Qwen linears        (--peft.target_modules regex, scoped to model.qwen)
#   - action head trained fully + saved     (model.action_model  -> full_training_modules)
#   - world model predictor trained + saved (model.video_predictor -> full_training_modules)
#   - 6-DoF action/state projections reinit'd from the 7-DoF checkpoint (--policy.reinit_modules);
#     they live inside model.action_model, so full_training_modules already trains them.
#
# !!! RUN scripts/smoke_test_vlajepa.sbatch FIRST !!!  It prints trainable-param names/counts so
# you can confirm Qwen adapters + action head + video_predictor are trainable BEFORE spending
# GPU-hours. If PEFT wrapping doesn't behave for this policy, fall back to full fine-tune
# (delete all --peft.* flags; keep everything else). See README "Fallback".
# =====================================================================================

# Train only on the train split by default. Compute the episode list HERE, inside the job — do NOT
# pass it via `sbatch --export=...,EPISODES=[0,1,2,...]`: Slurm's --export is comma-delimited, so a
# JSON list's commas get parsed as extra export entries and EPISODES arrives truncated to "[0",
# which then dies in draccus/yaml. Set TRAIN_SPLIT=0 to train on ALL 177 episodes instead.
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
