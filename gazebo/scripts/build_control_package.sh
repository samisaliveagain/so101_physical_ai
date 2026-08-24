#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  printf 'ROS 2 Jazzy was not found at /opt/ros/jazzy/setup.bash.\n' >&2
  exit 1
fi

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
export ROS_LOG_DIR="${PROJECT_ROOT}/.ros/log"
mkdir -p "${ROS_LOG_DIR}"

colcon --log-base "${PROJECT_ROOT}/log/so101_gazebo_control" build \
  --base-paths "${PROJECT_ROOT}/gazebo" \
  --build-base "${PROJECT_ROOT}/build/so101_gazebo_control" \
  --install-base "${PROJECT_ROOT}/install/so101_gazebo_control" \
  --packages-select so101_gazebo_control \
  --symlink-install \
  --cmake-args \
    -DPython3_EXECUTABLE=/usr/bin/python3 \
    -DPYTHON_EXECUTABLE=/usr/bin/python3
