"""Minimal SO-101 motor-bus setup based on the ECE 4560 joint-space example."""

from __future__ import annotations

import tomllib
from pathlib import Path

import draccus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "deployment" / "so101_hardware.toml"


def load_hardware_config(path: Path = DEFAULT_CONFIG) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_calibration(path: Path) -> dict[str, MotorCalibration]:
    with path.open() as handle, draccus.config_type("json"):
        return draccus.load(dict[str, MotorCalibration], handle)


def setup_bus(config: dict) -> FeetechMotorsBus:
    robot = config["robot"]
    bus = FeetechMotorsBus(
        port=robot["port"],
        motors={
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
        calibration=load_calibration(Path(robot["calibration"])),
    )
    bus.connect(handshake=True)
    if not bus.is_calibrated:
        bus.disconnect(disable_torque=False)
        raise RuntimeError("cached calibration does not match the connected SO-101")

    with bus.torque_disabled():
        bus.configure_motors()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            bus.write("P_Coefficient", motor, 16)
            bus.write("I_Coefficient", motor, 0)
            bus.write("D_Coefficient", motor, 32)
    return bus
