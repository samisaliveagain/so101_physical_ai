#!/usr/bin/env bash
set -eo pipefail

# Prevent Snap-packaged terminals from injecting Core20 glibc libraries into
# native ROS Jazzy / Gazebo transport commands.
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

usage() {
  cat <<'EOF'
Usage:
  send_calibrated_pose.sh PAN_DEG LIFT_DEG ELBOW_DEG WRIST_FLEX_DEG WRIST_ROLL_DEG GRIPPER_PERCENT [DURATION_SEC]

Example:
  ./gazebo/scripts/send_calibrated_pose.sh -18.77 -9.58 19.47 78.37 -5.23 9.05 4

The first five values use the calibrated LeRobot degree convention.
GRIPPER_PERCENT uses the dataset's 0..100 convention. DURATION_SEC must be an
integer from 1 through 30 and defaults to 4.
EOF
}

if (( $# < 6 || $# > 7 )); then
  usage >&2
  exit 2
fi

pan_deg="$1"
lift_deg="$2"
elbow_deg="$3"
wrist_flex_deg="$4"
wrist_roll_deg="$5"
gripper_percent="$6"
duration_sec="${7:-4}"

if [[ ! "${duration_sec}" =~ ^[0-9]+$ ]] || (( duration_sec < 1 || duration_sec > 30 )); then
  printf 'Rejected: DURATION_SEC must be an integer from 1 through 30.\n' >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
converted_output="$(python3 "${SCRIPT_DIR}/calibrated_to_urdf.py" \
  "${pan_deg}" "${lift_deg}" "${elbow_deg}" "${wrist_flex_deg}" \
  "${wrist_roll_deg}" "${gripper_percent}")" || exit $?
mapfile -t converted <<< "${converted_output}"
if (( ${#converted[@]} != 6 )); then
  printf 'Calibration converter returned %d values instead of six.\n' "${#converted[@]}" >&2
  exit 1
fi
pan_rad="${converted[0]}"
lift_rad="${converted[1]}"
elbow_rad="${converted[2]}"
wrist_flex_rad="${converted[3]}"
wrist_roll_rad="${converted[4]}"
gripper_rad="${converted[5]}"

controller_action='/arm_controller/follow_joint_trajectory'
controller_ready=false
for _attempt in {1..20}; do
  while IFS= read -r action; do
    if [[ "${action}" == "${controller_action}" ]]; then
      controller_ready=true
      break
    fi
  done < <(ros2 action list 2>/dev/null || true)
  [[ "${controller_ready}" == true ]] && break
  sleep 1
done

if [[ "${controller_ready}" != true ]]; then
  printf '%s\n' \
    "ROS action ${controller_action} was not discovered after 20 seconds." \
    'Confirm Gazebo was started with ./gazebo/scripts/run_world.sh.' >&2
  exit 1
fi

goal="{trajectory: {joint_names: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper], points: [{positions: [${pan_rad}, ${lift_rad}, ${elbow_rad}, ${wrist_flex_rad}, ${wrist_roll_rad}, ${gripper_rad}], time_from_start: {sec: ${duration_sec}, nanosec: 0}}]}}"

set +e
action_result="$(ros2 action send_goal \
  "${controller_action}" \
  control_msgs/action/FollowJointTrajectory \
  "${goal}" 2>&1)"
action_status=$?
set -e
printf '%s\n' "${action_result}"

if (( action_status != 0 )) || [[ "${action_result}" != *'Goal finished with status: SUCCEEDED'* ]]; then
  printf 'Trajectory did not complete successfully; the simulated arm was not reported at the goal.\n' >&2
  exit 1
fi

printf 'Trajectory succeeded: [%.2f, %.2f, %.2f, %.2f, %.2f] deg, gripper %.2f%% over %s s.\n' \
  "${pan_deg}" "${lift_deg}" "${elbow_deg}" "${wrist_flex_deg}" \
  "${wrist_roll_deg}" "${gripper_percent}" "${duration_sec}"
