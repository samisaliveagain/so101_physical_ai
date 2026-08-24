#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MOVE="${SCRIPT_DIR}/send_calibrated_pose.sh"

printf 'SO-101 calibration demo: three poses from real episode 0, about 30 seconds.\n'

# Real observation at t=0.5 s / frame 15. The shoulder value is clipped by
# 0.09 degrees to the recorded calibration endpoint. This is the folded start
# pose visible in hardware/camera_probe/training_left_t00_5.jpg.
"${MOVE}" -1.89011 -101.27472 91.47253 87.42857 2.32967 2.60452 6
sleep 2

# Real observation at t=5.0 s / frame 150: arm extended toward the nut.
"${MOVE}" -2.32967 32.35165 -51.03297 104.21978 18.85714 37.42289 8
sleep 2

# Real observation at t=15.0 s / frame 450: task-space adjustment above the nut.
"${MOVE}" -27.29670 35.95604 -57.53846 104.30769 22.02198 36.12063 5
sleep 3

printf 'Demo complete at the episode-0 t=15 s calibrated pose.\n'
