#!/usr/bin/env bash
# Convert the latest randomized LeRobot dataset, storing the result on the external drive.
set -euo pipefail

DRIVE_ROOT=${DRIVE_ROOT:-"/media/shubhamnagar/One Touch"}
LEROBOT_ROOT=${LEROBOT_ROOT:-/home/shubhamnagar/lerobot}
DATASET_ROOT=${DATASET_ROOT:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-}
CAMERA_KEY=${CAMERA_KEY:-observation.images.left}

if [[ -z "$DATASET_ROOT" ]]; then
  DATASET_ROOT=$(find "$DRIVE_ROOT" -maxdepth 1 -type d \
    -name 'so101_gazebo_randomized_stack_*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
fi
if [[ -z "$DATASET_ROOT" || ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  echo "Dataset not found. Set DATASET_ROOT." >&2
  exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_ROOT="$DRIVE_ROOT/so101_training"
OUTPUT_ROOT=${OUTPUT_ROOT:-"$TRAIN_ROOT/cosmos_data/so101_stack_$STAMP"}
SPLIT_FILE="$TRAIN_ROOT/splits/$(basename "$DATASET_ROOT")_seed42.json"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export HF_HOME="$TRAIN_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$TRAIN_ROOT/cache/xdg"
export TMPDIR="/tmp/so101_cosmos_convert_${UID}"
mkdir -p "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" "$(dirname "$SPLIT_FILE")" \
  "$(dirname "$OUTPUT_ROOT")"
chmod 700 "$TMPDIR"

WRITE_PROBE="$TRAIN_ROOT/.write_probe_$$"
if ! (umask 077 && : > "$WRITE_PROBE") 2>/dev/null; then
  echo "External drive is mounted read-only or has filesystem errors: $DRIVE_ROOT" >&2
  exit 1
fi
rm -f "$WRITE_PROBE"

"$LEROBOT_ROOT/.venv/bin/python" "$SCRIPT_DIR/prepare_episode_splits.py" \
  --dataset-root "$DATASET_ROOT" --output "$SPLIT_FILE"
exec "$LEROBOT_ROOT/.venv/bin/python" "$SCRIPT_DIR/convert_lerobot_to_cosmos.py" \
  --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT" \
  --split-file "$SPLIT_FILE" --camera-key "$CAMERA_KEY" "$@"
