#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORLD="${PROJECT_ROOT}/gazebo/worlds/gaussian_splat_stage1.sdf"
SPLAT="${PROJECT_ROOT}/3d_assets/gaussian_splats/Aachen-Mitte.ply"
CLEAN_SPLAT="${PROJECT_ROOT}/gazebo/gaussian/assets/Aachen-Mitte.cleaned.ply"
PREPARE_SPLAT="${PROJECT_ROOT}/gazebo/gaussian/tools/prepare_scaniverse_ply.py"
PORT="${SO101_GS_PORT:-8765}"
URL="http://127.0.0.1:${PORT}/gazebo/gaussian/viewer/"
HTTP_PID=""

cleanup() {
  if [[ -n "${HTTP_PID}" ]] && kill -0 "${HTTP_PID}" 2>/dev/null; then
    kill "${HTTP_PID}" 2>/dev/null || true
    wait "${HTTP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Keep the same Snap/Core20 isolation as the proven Gazebo launchers. Without
# this, a VS Code terminal can inject an incompatible libpthread / EGL stack.
if [[ -n "${SNAP:-}" ]]; then
  unset SNAP SNAP_ARCH SNAP_COMMON SNAP_CONTEXT SNAP_COOKIE SNAP_DATA
  unset SNAP_EUID SNAP_INSTANCE_NAME SNAP_LAUNCHER_ARCH_TRIPLET
  unset SNAP_LIBRARY_PATH SNAP_NAME SNAP_REAL_HOME SNAP_REVISION
  unset SNAP_UID SNAP_USER_COMMON SNAP_USER_DATA SNAP_VERSION
  unset GDK_PIXBUF_MODULEDIR GDK_PIXBUF_MODULE_FILE
  unset GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR GTK_EXE_PREFIX GTK_IM_MODULE_FILE
  unset GTK_PATH LOCPATH GI_TYPELIB_PATH FONTCONFIG_FILE FONTCONFIG_PATH
  unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QML2_IMPORT_PATH
  unset XDG_DATA_HOME
  export XDG_DATA_DIRS="/usr/local/share:/usr/share:/usr/share/ubuntu"
fi

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  printf 'ROS 2 Jazzy was not found at /opt/ros/jazzy/setup.bash.\n' >&2
  exit 1
fi
if [[ ! -f "${SPLAT}" ]]; then
  printf 'Gaussian splat was not found: %s\n' "${SPLAT}" >&2
  exit 1
fi

if [[ ! -f "${CLEAN_SPLAT}" || "${SPLAT}" -nt "${CLEAN_SPLAT}" ]]; then
  printf 'Preparing a finite, viewer-safe derivative of the Scaniverse PLY…\n'
  python3 "${PREPARE_SPLAT}" "${SPLAT}" "${CLEAN_SPLAT}"
fi

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# ROS / ament setup scripts are not nounset-safe. Enable strict undefined
# variable checking only after the environment has been loaded.
set -u

clean_colon_path() {
  local raw_path="${1:-}"
  local item cleaned=""
  local -a path_items=()
  IFS=':' read -r -a path_items <<< "${raw_path}"
  for item in "${path_items[@]}"; do
    [[ -z "${item}" || "${item}" == /snap/* ]] && continue
    cleaned="${cleaned:+${cleaned}:}${item}"
  done
  printf '%s' "${cleaned}"
}

export LD_LIBRARY_PATH="$(clean_colon_path "${LD_LIBRARY_PATH:-}")"
unset LD_PRELOAD
export ROS_LOG_DIR="${PROJECT_ROOT}/.ros/log"
mkdir -p "${ROS_LOG_DIR}"

python3 -m http.server "${PORT}" \
  --bind 127.0.0.1 \
  --directory "${PROJECT_ROOT}" \
  >"${PROJECT_ROOT}/.gaussian_stage1_http.log" 2>&1 &
HTTP_PID=$!

for _ in {1..30}; do
  if curl --silent --fail "${URL}" >/dev/null; then
    break
  fi
  sleep 0.1
done
if ! kill -0 "${HTTP_PID}" 2>/dev/null; then
  printf 'Local splat server failed. See %s/.gaussian_stage1_http.log\n' "${PROJECT_ROOT}" >&2
  exit 1
fi

printf 'Gaussian viewport: %s\n' "${URL}"
printf 'Gazebo stage-1 world: %s\n' "${WORLD}"
printf 'This stage intentionally has no plane, cameras, or robot.\n'

if command -v chromium >/dev/null 2>&1; then
  chromium --new-window "${URL}" >/dev/null 2>&1 &
elif command -v google-chrome >/dev/null 2>&1; then
  google-chrome --new-window "${URL}" >/dev/null 2>&1 &
else
  printf 'Open the Gaussian viewport URL in a WebGL2-capable browser.\n'
fi

gz sim -r "${WORLD}"
