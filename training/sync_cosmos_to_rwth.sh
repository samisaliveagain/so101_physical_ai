#!/usr/bin/env bash
# Copy this training bundle and a converted dataset to rwth-gpu without touching HPC $HOME.
set -euo pipefail

HOST=${HOST:-rwth-gpu}
REMOTE_ROOT=${REMOTE_ROOT:-/hpcwork/${HPC_USER:-$USER}/so101_cosmos}
LOCAL_DATASET=${LOCAL_DATASET:-}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ -z "$LOCAL_DATASET" || ! -f "$LOCAL_DATASET/manifest.json" ]]; then
  echo "Set LOCAL_DATASET to a converted Cosmos dataset (contains manifest.json)." >&2
  exit 2
fi

ssh "$HOST" "mkdir -p '$REMOTE_ROOT/bundle' '$REMOTE_ROOT/data/so101_stack' '$REMOTE_ROOT/logs'"
rsync -azh --info=progress2 "$SCRIPT_DIR/setup_cosmos_rwth.sh" \
  "$SCRIPT_DIR/train_cosmos_lora_rwth.sbatch" "$HOST:$REMOTE_ROOT/bundle/"
rsync -azh --info=progress2 "$LOCAL_DATASET/" "$HOST:$REMOTE_ROOT/data/so101_stack/"

echo "Synced. Next:"
echo "  ssh $HOST"
echo "  bash $REMOTE_ROOT/bundle/setup_cosmos_rwth.sh"
echo "  sbatch $REMOTE_ROOT/bundle/train_cosmos_lora_rwth.sbatch"
