#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Snap-packaged terminals (notably VS Code) export GTK / GIO paths from their
# Core20 runtime.  Passing those paths to a native Ubuntu 24.04 Gazebo process
# mixes two incompatible glibc versions and makes the GUI fail in libpthread.
# Remove only the inherited desktop-runtime variables for this child process.
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

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
fi

# ROS setup scripts may inspect unset variables, so enable nounset only after sourcing.
set -u

# Also discard any explicitly inherited Snap library entries while retaining
# all ROS Jazzy / Gazebo vendor directories added by setup.bash.
clean_colon_path() {
  local raw_path="${1:-}"
  local item
  local cleaned=""
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

# Build the resource package when it is missing or a source resource changed.
# The model contains many meshes, so an unchanged launch should not reinstall
# them every time.
overlay_setup="${PROJECT_ROOT}/install/so101_gazebo_control/setup.bash"
needs_build=false
if [[ ! -f "${overlay_setup}" ]]; then
  needs_build=true
elif [[ -n "$(find \
    "${PROJECT_ROOT}/gazebo/config" \
    "${PROJECT_ROOT}/gazebo/launch" \
    "${PROJECT_ROOT}/gazebo/models" \
    "${PROJECT_ROOT}/gazebo/rviz" \
    "${PROJECT_ROOT}/gazebo/worlds" \
    "${PROJECT_ROOT}/gazebo/CMakeLists.txt" \
    "${PROJECT_ROOT}/gazebo/package.xml" \
    -type f -newer "${overlay_setup}" -print -quit)" ]]; then
  needs_build=true
fi
if [[ "${needs_build}" == true ]]; then
  "${SCRIPT_DIR}/build_control_package.sh"
fi
# shellcheck disable=SC1091
set +u
source "${overlay_setup}"
set -u

# controlled_world.launch.py starts Gazebo, waits for the controller manager
# created by gz_ros2_control, and activates both the joint-state broadcaster
# and the six-joint trajectory controller.
exec ros2 launch so101_gazebo_control controlled_world.launch.py "gz_extra_args:=$*"
