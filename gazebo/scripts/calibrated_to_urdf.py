#!/usr/bin/env python3
"""Convert calibrated LeRobot SO-101 coordinates to URDF joint radians."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
# The CAD/URDF wrist-flex hinge axis is opposite to the positive direction of
# motor 4 in this follower's LeRobot observations. The other five axes agree.
URDF_DIRECTION = {
    "shoulder_pan": 1.0,
    "shoulder_lift": 1.0,
    "elbow_flex": 1.0,
    "wrist_flex": -1.0,
    "wrist_roll": 1.0,
}
ENCODER_MAX = 4095
GRIPPER_URDF_MIN = math.radians(-10.0)
GRIPPER_URDF_MAX = math.radians(100.0)


def default_calibration_path() -> Path:
    configured = os.environ.get("SO101_CALIBRATION_FILE")
    if configured:
        return Path(configured)
    live_candidate = (
        Path.home()
        / "calibration_lerobot_data_collect"
        / "calibration"
        / "robots"
        / "so_follower"
        / "my_awesome_follower_arm.json"
    )
    if live_candidate.is_file():
        return live_candidate
    source_candidate = Path(__file__).resolve().parents[1] / "config" / "so101_follower_calibration.json"
    if source_candidate.is_file():
        return source_candidate
    try:
        from ament_index_python.packages import get_package_share_directory

        return (
            Path(get_package_share_directory("so101_gazebo_control"))
            / "config"
            / "so101_follower_calibration.json"
        )
    except ImportError:
        return source_candidate


def calibrated_limits_degrees(calibration: dict, joint: str) -> tuple[float, float]:
    values = calibration[joint]
    if values["drive_mode"] != 0:
        raise ValueError(
            f"{joint}: drive_mode={values['drive_mode']} is not supported by this URDF mapping"
        )
    span = values["range_max"] - values["range_min"]
    if span <= 0:
        raise ValueError(f"{joint}: invalid encoder range {values['range_min']}..{values['range_max']}")
    half_range = (span / 2.0) * 360.0 / ENCODER_MAX
    return -half_range, half_range


def convert(values: list[float], calibration: dict) -> list[float]:
    if len(values) != 6:
        raise ValueError("expected five joint angles and one gripper percentage")

    result: list[float] = []
    for joint, degrees in zip(JOINTS, values[:5], strict=True):
        lower, upper = calibrated_limits_degrees(calibration, joint)
        if not lower <= degrees <= upper:
            raise ValueError(
                f"{joint}={degrees:.6f} deg is outside calibrated range "
                f"[{lower:.6f}, {upper:.6f}] deg"
            )
        # TheRobotStudio's so101_new_calib model and LeRobot both define zero
        # at the middle of the recorded joint range. No homing-offset term
        # belongs here: it is already applied by the physical motor firmware
        # before LeRobot normalization. Motor 4 is the one CAD-axis polarity
        # exception, handled explicitly above.
        result.append(URDF_DIRECTION[joint] * math.radians(degrees))

    gripper_percent = values[5]
    if not 0.0 <= gripper_percent <= 100.0:
        raise ValueError(f"gripper={gripper_percent:.6f}% is outside [0, 100]")
    result.append(
        GRIPPER_URDF_MIN
        + (gripper_percent / 100.0) * (GRIPPER_URDF_MAX - GRIPPER_URDF_MIN)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("values", type=float, nargs=6, metavar="VALUE")
    parser.add_argument("--calibration", type=Path, default=default_calibration_path())
    args = parser.parse_args()

    with args.calibration.open(encoding="utf-8") as handle:
        calibration = json.load(handle)

    missing = set((*JOINTS, "gripper")) - set(calibration)
    if missing:
        raise SystemExit(f"Calibration is missing motors: {', '.join(sorted(missing))}")

    try:
        converted = convert(args.values, calibration)
    except ValueError as exc:
        raise SystemExit(f"Rejected: {exc}") from exc

    for value in converted:
        print(f"{value:.12f}")


if __name__ == "__main__":
    main()
