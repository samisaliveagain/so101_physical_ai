#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LEROBOT_VENV=${LEROBOT_VENV:-"${HOME}/lerobot/.venv"}

if [[ ! -x "${LEROBOT_VENV}/bin/python" ]]; then
  printf 'LeRobot Python was not found at %s/bin/python\n' "${LEROBOT_VENV}" >&2
  exit 1
fi
if [[ -z "${SO101_ACT_CHECKPOINT:-}" ]]; then
  printf '%s\n' \
    'SO101_ACT_CHECKPOINT is required.' \
    'Set it to a LeRobot ACT pretrained_model directory.' >&2
  exit 2
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
