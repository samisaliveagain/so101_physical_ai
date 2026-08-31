#!/usr/bin/env bash
# Train ACT while keeping datasets, caches, and checkpoints under one storage root.
set -euo pipefail

DRIVE_ROOT=${DRIVE_ROOT:-"${SO101_STORAGE_ROOT:-${HOME}/so101_artifacts}"}
LEROBOT_ROOT=${LEROBOT_ROOT:-"${HOME}/lerobot"}
DATASET_ROOT=""
STEPS=100000
BATCH_SIZE=8
NUM_WORKERS=4
PUSH_TO_HUB=false
HF_REPO=""
RESUME_FROM=""
DRY_RUN=false

usage() {
  echo "Usage: $0 [--dataset PATH] [--steps N] [--batch-size N] [--resume RUN_OR_CHECKPOINT] [--push-to-hub [USER/REPO]] [--dry-run]"
}

while (($#)); do
  case "$1" in
    --dataset) DATASET_ROOT=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --batch-size) BATCH_SIZE=$2; shift 2 ;;
    --num-workers) NUM_WORKERS=$2; shift 2 ;;
    --resume) RESUME_FROM=$2; shift 2 ;;
    --push-to-hub)
      PUSH_TO_HUB=true
      if [[ ${2:-} != "" && ${2:-} != --* ]]; then HF_REPO=$2; shift 2; else shift; fi ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! mkdir -p "$DRIVE_ROOT"; then
  echo "Could not create storage root: $DRIVE_ROOT" >&2
  exit 1
fi
if [[ -z "$DATASET_ROOT" ]]; then
  DATASET_ROOT=$(find "$DRIVE_ROOT" -maxdepth 1 -type d \
    -name 'so101_gazebo_randomized_stack_*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
fi
if [[ -z "$DATASET_ROOT" || ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  echo "No LeRobot dataset found. Pass --dataset PATH." >&2
  exit 1
fi
if [[ ! -x "$LEROBOT_ROOT/.venv/bin/lerobot-train" ]]; then
  echo "Missing $LEROBOT_ROOT/.venv/bin/lerobot-train" >&2
  exit 1
fi

TRAIN_ROOT="$DRIVE_ROOT/so101_training"
export HF_HOME="$TRAIN_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$TRAIN_ROOT/cache/torch"
export XDG_CACHE_HOME="$TRAIN_ROOT/cache/xdg"
# Python multiprocessing uses Unix-domain sockets under TMPDIR. Keep these
# runtime files on the native Linux filesystem even when storage is removable.
export TMPDIR="/tmp/so101_training_${UID}"
export WANDB_DIR="$TRAIN_ROOT/wandb"
mkdir -p "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$WANDB_DIR" \
  "$TRAIN_ROOT/act" "$TRAIN_ROOT/splits"
chmod 700 "$TMPDIR"

WRITE_PROBE="$TRAIN_ROOT/.write_probe_$$"
if ! (umask 077 && : > "$WRITE_PROBE") 2>/dev/null; then
  echo "Storage root is read-only or has filesystem errors: $DRIVE_ROOT" >&2
  echo "Training cannot save caches/checkpoints until it is mounted read-write." >&2
  exit 1
fi
rm -f "$WRITE_PROBE"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATASET_NAME=$(basename "$DATASET_ROOT")
SPLIT_FILE="$TRAIN_ROOT/splits/${DATASET_NAME}_seed42.json"
"$LEROBOT_ROOT/.venv/bin/python" "$SCRIPT_DIR/prepare_episode_splits.py" \
  --dataset-root "$DATASET_ROOT" --output "$SPLIT_FILE"
TRAIN_EPISODES=$("$LEROBOT_ROOT/.venv/bin/python" -c \
  'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["train"], separators=(",",":")))' "$SPLIT_FILE")
TRAIN_EPISODE_COUNT=$("$LEROBOT_ROOT/.venv/bin/python" -c \
  'import json,sys; print(len(json.load(open(sys.argv[1]))["train"]))' "$SPLIT_FILE")

if [[ "$DRY_RUN" != true ]] && ! "$LEROBOT_ROOT/.venv/bin/python" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "CUDA is not visible in the LeRobot environment. Do not start ACT training on CPU." >&2
  echo "Check nvidia-smi and the CUDA-enabled torch installation." >&2
  exit 1
fi

PORTABLE_TRAIN="$SCRIPT_DIR/lerobot_train_portable.py"
if [[ -n "$RESUME_FROM" ]]; then
  if [[ -f "$RESUME_FROM" && $(basename "$RESUME_FROM") == train_config.json ]]; then
    CONFIG_PATH=$RESUME_FROM
    CHECKPOINT_DIR=$(dirname -- "$(dirname -- "$CONFIG_PATH")")
  elif [[ -f "$RESUME_FROM/pretrained_model/train_config.json" ]]; then
    CHECKPOINT_DIR=$RESUME_FROM
    CONFIG_PATH="$CHECKPOINT_DIR/pretrained_model/train_config.json"
  elif [[ -d "$RESUME_FROM/checkpoints" ]]; then
    CHECKPOINT_DIR=$(find "$RESUME_FROM/checkpoints" -mindepth 1 -maxdepth 1 -type d \
      -name '[0-9]*' -printf '%f %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    if [[ -z "$CHECKPOINT_DIR" ]]; then
      echo "No numbered checkpoint found under: $RESUME_FROM/checkpoints" >&2
      exit 1
    fi
    CONFIG_PATH="$CHECKPOINT_DIR/pretrained_model/train_config.json"
  else
    echo "--resume expects a run directory, checkpoint directory, or train_config.json: $RESUME_FROM" >&2
    exit 1
  fi

  if [[ ! -f "$CONFIG_PATH" || ! -f "$CHECKPOINT_DIR/training_state/training_step.json" ]]; then
    echo "Incomplete checkpoint: $CHECKPOINT_DIR" >&2
    exit 1
  fi

  RESUME_STEP=$("$LEROBOT_ROOT/.venv/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["step"])' \
    "$CHECKPOINT_DIR/training_state/training_step.json")
  if ((STEPS <= RESUME_STEP)); then
    echo "--steps is the final global step and must exceed checkpoint step $RESUME_STEP." >&2
    exit 1
  fi
  OUTPUT_DIR=$(dirname -- "$(dirname -- "$CHECKPOINT_DIR")")
  RUN_ID=$(basename "$OUTPUT_DIR")
  CMD=("$LEROBOT_ROOT/.venv/bin/python" "$PORTABLE_TRAIN"
    --config_path="$CONFIG_PATH"
    --resume=true
    --dataset.repo_id=local/so101_gazebo_act_training
    --dataset.root="$DATASET_ROOT"
    --dataset.episodes="$TRAIN_EPISODES"
    --dataset.video_backend=pyav
    --steps="$STEPS"
    --batch_size="$BATCH_SIZE"
    --num_workers="$NUM_WORKERS")
else
  RUN_ID="act_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)"
  OUTPUT_DIR="$TRAIN_ROOT/act/$RUN_ID"
  CMD=("$LEROBOT_ROOT/.venv/bin/python" "$PORTABLE_TRAIN"
    --dataset.repo_id=local/so101_gazebo_randomized_stack
    --dataset.root="$DATASET_ROOT"
    --dataset.episodes="$TRAIN_EPISODES"
    --dataset.video_backend=pyav
    --policy.type=act
    --policy.device=cuda
    --policy.pretrained_backbone_weights=null
    --policy.input_features='{"observation.state":{"type":"STATE","shape":[6]},"observation.images.left":{"type":"VISUAL","shape":[3,480,640]},"observation.images.fpv":{"type":"VISUAL","shape":[3,480,640]}}'
    --policy.push_to_hub="$PUSH_TO_HUB"
    --output_dir="$OUTPUT_DIR"
    --job_name="$RUN_ID"
    --batch_size="$BATCH_SIZE"
    --steps="$STEPS"
    --num_workers="$NUM_WORKERS"
    --save_checkpoint=true
    --save_freq=10000
    --log_freq=100
    --wandb.enable=false)
fi

if [[ "$PUSH_TO_HUB" == true ]]; then
  hf auth whoami >/dev/null
  if [[ -z "$HF_REPO" ]]; then
    HF_USER=$(hf auth whoami --format json | "$LEROBOT_ROOT/.venv/bin/python" -c \
      'import json,sys; print(json.load(sys.stdin)["user"])')
    HF_REPO="$HF_USER/so101-act-nut-stack"
  fi
  CMD+=(--policy.repo_id="$HF_REPO")
fi

echo "Dataset: $DATASET_ROOT"
echo "Split:   $SPLIT_FILE ($TRAIN_EPISODE_COUNT train episodes)"
echo "Output:  $OUTPUT_DIR"
if [[ -n "$RESUME_FROM" ]]; then
  echo "Resume:  $CHECKPOINT_DIR (completed step $RESUME_STEP)"
fi
printf 'Command:'; printf ' %q' "${CMD[@]}"; printf '\n'
if [[ "$DRY_RUN" == true ]]; then exit 0; fi
exec "${CMD[@]}"
