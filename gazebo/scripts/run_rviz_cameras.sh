#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Prevent Snap-packaged terminals from injecting their Core20 glibc / GUI
# libraries into native ROS Jazzy and RViz processes.
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
set -u

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

# RViz resolves the URDF's package:// mesh URIs through the ament index. A
# separate terminal only knows the Jazzy underlay until this project overlay is
# sourced, even when Gazebo was launched successfully from another terminal.
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
set +u
# shellcheck disable=SC1090
source "${overlay_setup}"
set -u

if ! package_share="$(ros2 pkg prefix --share so101_gazebo_control 2>/dev/null)"; then
  printf 'RViz cannot resolve package://so101_gazebo_control. Rebuild the project overlay.\n' >&2
  exit 1
fi
robot_urdf="${package_share}/models/so101_dark_blue/so101_dark_blue.urdf"
if [[ ! -r "${robot_urdf}" ]]; then
  printf 'Installed robot URDF is missing: %s\n' "${robot_urdf}" >&2
  exit 1
fi

bridge_pid=''
cleanup() {
  if [[ -n "${bridge_pid}" ]]; then
    kill "${bridge_pid}" 2>/dev/null || true
    wait "${bridge_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

left_camera_ready=false
fpv_camera_ready=false
for _attempt in {1..10}; do
  while IFS= read -r topic; do
    [[ "${topic}" == '/so101/camera/left/image' ]] && left_camera_ready=true
    [[ "${topic}" == '/so101/camera/fpv/image' ]] && fpv_camera_ready=true
  done < <(gz topic -l 2>/dev/null || true)
  if [[ "${left_camera_ready}" == true && "${fpv_camera_ready}" == true ]]; then
    break
  fi
  sleep 1
done

if [[ "${left_camera_ready}" != true || "${fpv_camera_ready}" != true ]]; then
  printf '%s\n' \
    'Gazebo camera topics were not discovered after 10 seconds.' \
    'Start ./gazebo/scripts/run_world.sh before launching RViz.' >&2
  exit 1
fi

robot_description_ready=false
for _attempt in {1..10}; do
  while IFS= read -r topic; do
    if [[ "${topic}" == '/robot_description' ]]; then
      robot_description_ready=true
      break
    fi
  done < <(ros2 topic list 2>/dev/null || true)
  [[ "${robot_description_ready}" == true ]] && break
  sleep 1
done
if [[ "${robot_description_ready}" != true ]]; then
  printf '%s\n' \
    'ROS topic /robot_description was not discovered after 10 seconds.' \
    'Start ./gazebo/scripts/run_world.sh in the same ROS_DOMAIN_ID first.' >&2
  exit 1
fi

ros2 run ros_gz_bridge parameter_bridge \
  '/so101/camera/left/image@sensor_msgs/msg/Image[gz.msgs.Image' \
  '/so101/camera/left/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo' \
  '/so101/camera/fpv/image@sensor_msgs/msg/Image[gz.msgs.Image' \
  '/so101/camera/fpv/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo' &
bridge_pid="$!"

sleep 2
if ! kill -0 "${bridge_pid}" 2>/dev/null; then
  printf 'The ROS-Gazebo bridge exited. Make sure Gazebo is already running.\n' >&2
  exit 1
fi

rviz2 -d "${PROJECT_ROOT}/gazebo/rviz/so101_cameras.rviz"
