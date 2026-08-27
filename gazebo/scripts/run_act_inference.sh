#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LEROBOT_VENV=${LEROBOT_VENV:-/home/shubhamnagar/lerobot/.venv}
DEFAULT_CHECKPOINT='/media/shubhamnagar/One Touch/so101_training/act/act_so101_gazebo_randomized_stack_20260824_005251_20260824_182755/checkpoints/100000/pretrained_model'
export SO101_ACT_CHECKPOINT=${SO101_ACT_CHECKPOINT:-"${DEFAULT_CHECKPOINT}"}

if [[ ! -x "${LEROBOT_VENV}/bin/python" ]]; then
  printf 'LeRobot Python was not found at %s/bin/python\n' "${LEROBOT_VENV}" >&2
  exit 1
fi
if [[ ! -f "${SO101_ACT_CHECKPOINT}/model.safetensors" ]]; then
  printf 'ACT checkpoint was not found at %s\n' "${SO101_ACT_CHECKPOINT}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
export PYTHONUNBUFFERED=1

exec "${LEROBOT_VENV}/bin/python" \
  "${PROJECT_ROOT}/gazebo/scripts/so101_act_gazebo_bridge.py" \
  --checkpoint "${SO101_ACT_CHECKPOINT}" "$@"
