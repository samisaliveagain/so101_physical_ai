#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${SO101_DATA_ROOT:-${DRIVE_ROOT:-${HOME}/so101_data}}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT="${DATA_ROOT}/so101_gazebo_randomized_stack_${RUN_STAMP}"
SIM_LOG="${PROJECT_ROOT}/.ros/headless_collection_${RUN_STAMP}.log"

if ! mkdir -p "${DATA_ROOT}"; then
  printf 'Could not create data directory: %s\n' "${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -w "${DATA_ROOT}" ]]; then
  printf 'Data directory is not writable: %s\n' "${DATA_ROOT}" >&2
  exit 1
fi

has_output=false
for argument in "$@"; do
  if [[ "${argument}" == "--output" || "${argument}" == --output=* ]]; then
    has_output=true
    break
  fi
done

mkdir -p "${PROJECT_ROOT}/.ros"
setsid "${SCRIPT_DIR}/run_sim_rviz.sh" \
  launch_rviz:=false \
  'gz_extra_args:=-s --headless-rendering' >"${SIM_LOG}" 2>&1 &
sim_pid=$!

cleanup() {
  if kill -0 "${sim_pid}" 2>/dev/null; then
    kill -TERM -- "-${sim_pid}" 2>/dev/null || true
    wait "${sim_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# The collector itself waits up to 15 seconds for controllers and both cameras.
if [[ "${has_output}" == true ]]; then
  "${SCRIPT_DIR}/collect_randomized_lerobot.sh" "$@"
else
  "${SCRIPT_DIR}/collect_randomized_lerobot.sh" --output "${DEFAULT_OUTPUT}" "$@"
fi

printf 'Headless collection finished. Simulation log: %s\n' "${SIM_LOG}"
