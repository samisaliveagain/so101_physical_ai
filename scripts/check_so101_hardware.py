#!/usr/bin/env python3
"""Non-actuating validation for the SO-101 deployment hardware mapping."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "deployment" / "so101_hardware.toml"
EXPECTED_MOTORS = {
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
}
CALIBRATION_FIELDS = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def check_calibration(path: Path) -> None:
    calibration = json.loads(path.read_text())
    if set(calibration) != EXPECTED_MOTORS:
        raise RuntimeError(f"Unexpected calibration motors: {sorted(calibration)}")
    for motor, values in calibration.items():
        missing = CALIBRATION_FIELDS - set(values)
        if missing:
            raise RuntimeError(f"Calibration for {motor} is missing {sorted(missing)}")
    print(f"calibration: OK ({path}, six motors)")


def check_camera(label: str, config: dict) -> None:
    device = Path(config["device"])
    if not device.exists():
        raise RuntimeError(f"{label}: camera path does not exist: {device}")

    capture = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
    try:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config["fourcc"]))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
        capture.set(cv2.CAP_PROP_FPS, config["fps"])
        ok, frame = capture.read()
        if not capture.isOpened() or not ok or frame is None:
            raise RuntimeError(f"{label}: failed to capture a frame from {device}")
        height, width = frame.shape[:2]
        print(
            f"camera {label}: OK ({width}x{height}, {config['dataset_key']} -> "
            f"{config['policy_key']})"
        )
    finally:
        capture.release()


def read_motor_state(config: dict) -> None:
    """Ping motors and read calibration/position registers; never write registers."""
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    robot_config = SOFollowerRobotConfig(
        port=config["port"],
        id=config["id"],
        cameras={},
    )
    robot = SOFollower(robot_config)
    robot.bus.connect(handshake=True)
    try:
        matches_cache = robot.bus.is_calibrated
        positions = robot.bus.sync_read("Present_Position")
        print(f"motors: OK (IDs 1-6, calibration_matches_hardware={matches_cache})")
        print("positions:", ", ".join(f"{name}={value:.2f}" for name, value in positions.items()))
    finally:
        # False is intentional: disabling torque is a register write. This checker sends no writes.
        robot.bus.disconnect(disable_torque=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--read-motors",
        action="store_true",
        help="Open the serial bus and read motor calibration/positions without writing registers.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    check_calibration(Path(config["robot"]["calibration"]))
    for label, camera_config in config["cameras"].items():
        check_camera(label, camera_config)
    if args.read_motors:
        read_motor_state(config["robot"])
    else:
        print("motors: skipped (pass --read-motors for a read-only live check)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"hardware check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
