#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/jazzy/setup.bash
export LEROBOT_PYTHON=${LEROBOT_PYTHON:-/opt/lerobot/bin/python}
export DRIVE_ROOT=${DRIVE_ROOT:-/datasets}
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-71}
export GZ_IP=127.0.0.1

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/so101}
cd "$PROJECT_ROOT"
exec gazebo/scripts/collect_randomized_lerobot_headless.sh "$@"
