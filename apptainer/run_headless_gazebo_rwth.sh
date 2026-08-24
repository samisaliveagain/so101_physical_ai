#!/usr/bin/env bash
# Run the SO-101 randomized collector inside its immutable Apptainer image.
set -euo pipefail

HPC_ROOT=${HPC_ROOT:-"/hpcwork/$USER/so101_gazebo"}
PROJECT_ROOT=${PROJECT_ROOT:-"$HPC_ROOT/project"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$HPC_ROOT/datasets"}
IMAGE=${IMAGE:-"$HPC_ROOT/images/so101_gazebo_jazzy.sif"}
EPISODES=${EPISODES:-10}
SEED=${SEED:-101}
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_NAME=${OUTPUT_NAME:-"so101_gazebo_apptainer_${RUN_STAMP}"}

if [[ ! -s "$IMAGE" ]]; then
  echo "Apptainer image missing: $IMAGE" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT" "$HPC_ROOT/tmp" "$HPC_ROOT/ros_logs"

APPTAINER_ARGS=(--cleanenv --writable-tmpfs
  --bind "$PROJECT_ROOT:/workspace/so101:rw"
  --bind "$OUTPUT_ROOT:/datasets:rw"
  --bind "$HPC_ROOT/tmp:/runtime_tmp:rw")
if [[ ${USE_NVIDIA:-1} == 1 ]]; then APPTAINER_ARGS+=(--nv); fi

export APPTAINERENV_PROJECT_ROOT=/workspace/so101
export APPTAINERENV_DRIVE_ROOT=/datasets
export APPTAINERENV_LEROBOT_PYTHON=/opt/lerobot/bin/python
export APPTAINERENV_ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-71}
export APPTAINERENV_ROS_LOG_DIR=/runtime_tmp/ros_logs
export APPTAINERENV_TMPDIR=/runtime_tmp
export APPTAINERENV_QT_QPA_PLATFORM=offscreen
if [[ ${SOFTWARE_RENDERING:-0} == 1 ]]; then
  export APPTAINERENV_LIBGL_ALWAYS_SOFTWARE=1
  export APPTAINERENV_MESA_GL_VERSION_OVERRIDE=4.5
fi

exec apptainer exec "${APPTAINER_ARGS[@]}" "$IMAGE" \
  /workspace/so101/apptainer/container_headless_entrypoint.sh \
  --episodes "$EPISODES" --seed "$SEED" --output "/datasets/$OUTPUT_NAME" "$@"
