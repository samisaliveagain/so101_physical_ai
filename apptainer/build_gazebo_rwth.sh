#!/usr/bin/env bash
# Build once on rwth-gpu. The image and Apptainer cache live under /hpcwork.
set -euo pipefail

HPC_ROOT=${HPC_ROOT:-"/hpcwork/$USER/so101_gazebo"}
PROJECT_ROOT=${PROJECT_ROOT:-"$HPC_ROOT/project"}
DEFINITION=${DEFINITION:-"$PROJECT_ROOT/apptainer/so101_gazebo_jazzy.def"}
IMAGE=${IMAGE:-"$HPC_ROOT/images/so101_gazebo_jazzy.sif"}

export APPTAINER_CACHEDIR="$HPC_ROOT/cache/apptainer"
export APPTAINER_TMPDIR="$HPC_ROOT/tmp"
mkdir -p "$HPC_ROOT/images" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

if [[ -s "$IMAGE" ]]; then
  echo "Image already exists: $IMAGE"
  apptainer test "$IMAGE"
  exit 0
fi
if [[ ! -f "$DEFINITION" ]]; then
  echo "Definition not found: $DEFINITION" >&2
  exit 1
fi

PARTIAL="$HPC_ROOT/images/so101_gazebo_jazzy.partial.sif"
apptainer build --fakeroot "$PARTIAL" "$DEFINITION"
mv "$PARTIAL" "$IMAGE"
apptainer test "$IMAGE"
echo "Built $IMAGE"
