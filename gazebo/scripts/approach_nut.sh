#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
if [[ -f "${PROJECT_ROOT}/install/so101_gazebo_control/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${PROJECT_ROOT}/install/so101_gazebo_control/setup.bash"
fi
set -u

# ROS Jazzy is built for Ubuntu's Python 3.12. An active Conda environment may
# put an ABI-incompatible Python ahead of it on PATH.
exec /usr/bin/python3 "${SCRIPT_DIR}/approach_nut.py" \
  --urdf "${PROJECT_ROOT}/gazebo/models/so101_dark_blue/so101_dark_blue.urdf" "$@"
