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

LEROBOT_PYTHON="${LEROBOT_PYTHON:-/home/shubhamnagar/lerobot/.venv/bin/python}"
if [[ ! -x "${LEROBOT_PYTHON}" ]]; then
  printf 'LeRobot Python not found at %s\n' "${LEROBOT_PYTHON}" >&2
  exit 1
fi
exec "${LEROBOT_PYTHON}" "${SCRIPT_DIR}/collect_randomized_lerobot.py" "$@"
