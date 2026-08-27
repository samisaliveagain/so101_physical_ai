#!/usr/bin/env bash
# Start a new ACT run with pretrained vision and reduced stationary-frame sampling.
set -euo pipefail

DRIVE_ROOT=${DRIVE_ROOT:-"/media/shubhamnagar/One Touch"}
LEROBOT_ROOT=${LEROBOT_ROOT:-/home/shubhamnagar/lerobot}
DATASET_ROOT="${DRIVE_ROOT}/so101_gazebo_combined_verified_100_20260826"
STEPS=100000
BATCH_SIZE=8
NUM_WORKERS=4
CHUNK_SIZE=50
ACTION_HORIZON=10
STATIC_THRESHOLD_RAD=0.0001
MAX_STATIC_FRAMES=15
FINAL_STATIC_FRAMES=5
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: train_act_pretrained_trimmed.sh [OPTIONS]

Starts a completely new ACT run. Resuming an older checkpoint is intentionally
not supported.

Options:
  --dataset PATH             LeRobot dataset (default: combined verified 100)
  --steps N                  Training updates (default: 100000)
  --batch-size N             Batch size (default: 8)
  --num-workers N            Data-loader workers (default: 4)
  --chunk-size N             Predicted ACT chunk (default: 100)
  --action-horizon N         Actions executed before replanning (default: 10)
  --static-threshold RAD     Per-step stationary threshold (default: 0.0001)
  --max-static-frames N      Maximum sampled internal hold (default: 15)
  --dry-run                  Print the command without starting training
  -h, --help                 Show this help
EOF
}

while (($#)); do
  case "$1" in
    --dataset) DATASET_ROOT=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --batch-size) BATCH_SIZE=$2; shift 2 ;;
    --num-workers) NUM_WORKERS=$2; shift 2 ;;
    --chunk-size) CHUNK_SIZE=$2; shift 2 ;;
    --action-horizon) ACTION_HORIZON=$2; shift 2 ;;
    --static-threshold) STATIC_THRESHOLD_RAD=$2; shift 2 ;;
    --max-static-frames) MAX_STATIC_FRAMES=$2; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --resume|--resume=*)
      echo "This launcher always trains a new policy; --resume is not allowed." >&2
      exit 2
      ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in "$STEPS" "$BATCH_SIZE" "$CHUNK_SIZE" "$ACTION_HORIZON" "$MAX_STATIC_FRAMES"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Steps, batch, chunk, horizon and static-frame values must be positive integers." >&2
    exit 2
  fi
done
if ((ACTION_HORIZON > CHUNK_SIZE)); then
  echo "--action-horizon cannot exceed --chunk-size." >&2
  exit 2
fi
if [[ ! "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "--num-workers must be a non-negative integer." >&2
  exit 2
fi

if [[ ! -d "$DRIVE_ROOT" || ! -w "$DRIVE_ROOT" ]]; then
  echo "External drive is not mounted read-write at: $DRIVE_ROOT" >&2
  exit 1
fi
if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  echo "LeRobot dataset was not found at: $DATASET_ROOT" >&2
  exit 1
fi
if [[ ! -x "$LEROBOT_ROOT/.venv/bin/python" ]]; then
  echo "LeRobot Python is missing: $LEROBOT_ROOT/.venv/bin/python" >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAIN_ROOT="$DRIVE_ROOT/so101_training"
DATASET_NAME=$(basename "$DATASET_ROOT")
RUN_ID="act_pretrained_trimmed_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$TRAIN_ROOT/act/$RUN_ID"
SPLIT_FILE="$TRAIN_ROOT/splits/${DATASET_NAME}_seed42.json"
TRIM_REPORT="$TRAIN_ROOT/splits/${RUN_ID}_static_trim.json"

# Keep model downloads, checkpoints, training logs and bulky caches off the
# system disk. Only Unix sockets and locks use the native /tmp filesystem.
export HF_HOME="$TRAIN_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$TRAIN_ROOT/cache/torch"
export XDG_CACHE_HOME="$TRAIN_ROOT/cache/xdg"
export WANDB_DIR="$TRAIN_ROOT/wandb"
export TMPDIR="/tmp/so101_act_pretrained_${UID}"
export SO101_STATIC_THRESHOLD_RAD="$STATIC_THRESHOLD_RAD"
export SO101_MAX_STATIC_FRAMES="$MAX_STATIC_FRAMES"
export SO101_FINAL_STATIC_FRAMES="$FINAL_STATIC_FRAMES"
export SO101_STATIC_TRIM_REPORT="$TRIM_REPORT"
mkdir -p "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$WANDB_DIR" \
  "$TMPDIR" "$TRAIN_ROOT/act" "$TRAIN_ROOT/splits"
chmod 700 "$TMPDIR"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite an existing training run: $OUTPUT_DIR" >&2
  exit 1
fi

"$LEROBOT_ROOT/.venv/bin/python" "$SCRIPT_DIR/prepare_episode_splits.py" \
  --dataset-root "$DATASET_ROOT" --output "$SPLIT_FILE"
TRAIN_EPISODES=$("$LEROBOT_ROOT/.venv/bin/python" -c \
  'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["train"], separators=(",",":")))' \
  "$SPLIT_FILE")
TRAIN_EPISODE_COUNT=$("$LEROBOT_ROOT/.venv/bin/python" -c \
  'import json,sys; print(len(json.load(open(sys.argv[1]))["train"]))' "$SPLIT_FILE")

if [[ "$DRY_RUN" != true ]] && ! "$LEROBOT_ROOT/.venv/bin/python" -c \
  'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "CUDA is not visible in the LeRobot environment; refusing CPU training." >&2
  exit 1
fi

CMD=("$LEROBOT_ROOT/.venv/bin/python" "$SCRIPT_DIR/lerobot_train_static_trimmed.py"
  --dataset.repo_id=local/so101_gazebo_pretrained_trimmed
  --dataset.root="$DATASET_ROOT"
  --dataset.episodes="$TRAIN_EPISODES"
  --dataset.video_backend=pyav
  --dataset.use_imagenet_stats=true
  --policy.type=act
  --policy.device=cuda
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
  --policy.use_vae=true
  --policy.chunk_size="$CHUNK_SIZE"
  --policy.n_action_steps="$ACTION_HORIZON"
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[6]},"observation.images.left":{"type":"VISUAL","shape":[3,480,640]},"observation.images.fpv":{"type":"VISUAL","shape":[3,480,640]}}'
  --policy.push_to_hub=false
  --output_dir="$OUTPUT_DIR"
  --job_name="$RUN_ID"
  --batch_size="$BATCH_SIZE"
  --steps="$STEPS"
  --num_workers="$NUM_WORKERS"
  --save_checkpoint=true
  --save_freq=10000
  --log_freq=100
  --wandb.enable=false)

echo "Fresh ACT training run (no checkpoint resume)"
echo "Dataset:       $DATASET_ROOT"
echo "Split:         $SPLIT_FILE ($TRAIN_EPISODE_COUNT train episodes)"
echo "Output:        $OUTPUT_DIR"
echo "Trim report:   $TRIM_REPORT"
echo "Vision:        ImageNet-pretrained ResNet-18, then fine-tuned"
echo "ACT:           VAE=true, chunk=$CHUNK_SIZE, action horizon=$ACTION_HORIZON"
echo "Static cap:    $MAX_STATIC_FRAMES frames at threshold $STATIC_THRESHOLD_RAD rad"
printf 'Command:'; printf ' %q' "${CMD[@]}"; printf '\n'
if [[ "$DRY_RUN" == true ]]; then
  exit 0
fi

exec "${CMD[@]}"
