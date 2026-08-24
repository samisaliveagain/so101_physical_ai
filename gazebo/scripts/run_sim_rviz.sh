#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Prevent Snap-packaged terminals from injecting Core20 desktop libraries into
# native ROS Jazzy, Gazebo and RViz processes.
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

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
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

"${SCRIPT_DIR}/build_control_package.sh"
set +u
# shellcheck disable=SC1090
source "${PROJECT_ROOT}/install/so101_gazebo_control/setup.bash"
set -u

# Two robot_state_publisher/controller-manager instances on one ROS domain
# publish competing TF and joint data, which looks exactly like RViz lag or a
# static robot.  Refuse that ambiguous state and tell the user how to recover.
existing_sim=false
while IFS= read -r node_name; do
  case "${node_name}" in
    /controller_manager|/robot_state_publisher)
      existing_sim=true
      break
      ;;
  esac
done < <(ros2 node list 2>/dev/null || true)
if [[ "${existing_sim}" == true ]]; then
  printf '%s\n' \
    'An SO-101/ROS simulation appears to be running in this ROS domain.' \
    'Stop the old Gazebo and RViz launch (Ctrl+C), then run this command again.' >&2
  exit 1
fi

exec ros2 launch so101_gazebo_control sim_rviz.launch.py "$@"
