#!/usr/bin/env python3
"""Physically verify SO-101 goal-position control without cameras, HTTP, or a policy."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
import time
import tomllib
from pathlib import Path
from threading import Event

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deployment"))

from so101_http_slow_rollout import execute_and_verify  # noqa: E402
from so101_safety import SafetyLimits, plan_quintic_trajectory  # noqa: E402


DEFAULT_CONFIG = ROOT / "deployment" / "so101_hardware.toml"
DEFAULT_LOG = ROOT / "eval" / "out" / "direct_motion_test.jsonl"
ARM_PHRASE = "MOVE_SO101_TEST"
LOCK_PATH = Path("/tmp/so101_http_slow_rollout.lock")


class DirectMotorHardware:
    def __init__(self, robot_config: dict, limits: SafetyLimits):
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.robots.so_follower.so_follower import SOFollower

        relative_limit = dict(zip(limits.joint_names, limits.max_chunk_delta.tolist(), strict=True))
        self.joint_names = limits.joint_names
        self.robot = SOFollower(
            SOFollowerRobotConfig(
                port=robot_config["port"],
                id=robot_config["id"],
                cameras={},
                use_degrees=True,
                max_relative_target=relative_limit,
                disable_torque_on_disconnect=False,
            )
        )

        # Read-only calibration/state gate before connect() configures the motors.
        self.robot.bus.connect(handshake=True)
        try:
            if not self.robot.bus.is_calibrated:
                raise RuntimeError("cached calibration does not match the connected arm")
            positions = self.robot.bus.sync_read("Present_Position")
            state = np.asarray([positions[name] for name in self.joint_names], dtype=np.float64)
            if np.any(state < limits.joint_min) or np.any(state > limits.joint_max):
                raise RuntimeError("current state lies outside the configured demonstrated envelope")
        finally:
            self.robot.bus.disconnect(disable_torque=False)

        self.robot.connect(calibrate=False)
        torque = self.robot.bus.sync_read("Torque_Enable")
        if not all(int(value) == 1 for value in torque.values()):
            self.robot.bus.enable_torque()
            torque = self.robot.bus.sync_read("Torque_Enable")
        if not all(int(value) == 1 for value in torque.values()):
            self.robot.disconnect()
            raise RuntimeError("one or more motors could not be torque-enabled")

    def state(self) -> np.ndarray:
        positions = self.robot.bus.sync_read("Present_Position")
        return np.asarray([positions[name] for name in self.joint_names], dtype=np.float64)

    def command(self, positions: np.ndarray) -> np.ndarray:
        action = {
            f"{name}.pos": float(value)
            for name, value in zip(self.joint_names, positions, strict=True)
        }
        sent = self.robot.send_action(action)
        return np.asarray([sent[f"{name}.pos"] for name in self.joint_names], dtype=np.float64)

    def hold(self) -> np.ndarray:
        current = self.state()
        self.command(current)
        return current

    def close(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()


def audit(path: Path, event: str, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"unix_ms": time.time_ns() / 1_000_000, "event": event, **fields},
                separators=(",", ":"),
            )
            + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--joint", default="wrist_roll")
    parser.add_argument("--delta", type=float, default=3.0)
    parser.add_argument("--return-to-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arm", default="")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    if args.arm != ARM_PHRASE:
        raise RuntimeError(f"motion is locked; pass --arm {ARM_PHRASE}")
    if not sys.stdin.isatty():
        raise RuntimeError("refusing physical motion without an interactive terminal")

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    limits = SafetyLimits.from_config(config["safety"])
    if args.joint not in limits.joint_names:
        raise ValueError(f"unknown joint {args.joint!r}; choose from {limits.joint_names}")
    joint_index = limits.joint_names.index(args.joint)
    if not np.isfinite(args.delta) or abs(args.delta) < limits.minimum_commanded_motion[joint_index]:
        raise ValueError("motion delta is too small to verify")
    if abs(args.delta) > limits.max_model_delta[joint_index]:
        raise ValueError(
            f"motion delta exceeds the configured cap of {limits.max_model_delta[joint_index]:.3f}"
        )

    print(
        f"This will move {args.joint} by {args.delta:+.2f} units"
        + (" and return to its start position." if args.return_to_start else ".")
    )
    print("Keep the physical e-stop/power cut in reach.")
    confirmation = input(f"Type {ARM_PHRASE} to continue: ")
    if confirmation != ARM_PHRASE:
        raise RuntimeError("arming confirmation did not match")

    stop = Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())

    hardware: DirectMotorHardware | None = None
    with LOCK_PATH.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another SO-101 rollout or motion test is already running") from exc
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        try:
            hardware = DirectMotorHardware(config["robot"], limits)
            start = hardware.hold()
            target = start.copy()
            target[joint_index] += args.delta
            if not limits.joint_min[joint_index] <= target[joint_index] <= limits.joint_max[joint_index]:
                raise RuntimeError("requested target lies outside the configured joint envelope")

            audit(args.log, "outbound_started", joint=args.joint, start=start.tolist(), target=target.tolist())
            outbound = plan_quintic_trajectory(start, target, limits)
            reached, outbound_tracking = execute_and_verify(
                hardware, start, target, outbound, limits, stop
            )
            audit(
                args.log,
                "outbound_complete",
                measured=reached.tolist(),
                tracking=outbound_tracking,
            )
            print(f"Outbound movement verified: {args.joint}={reached[joint_index]:.2f}", flush=True)

            final = reached
            return_tracking = None
            if args.return_to_start:
                inbound = plan_quintic_trajectory(reached, start, limits)
                final, return_tracking = execute_and_verify(
                    hardware, reached, start, inbound, limits, stop
                )
                audit(
                    args.log,
                    "return_complete",
                    measured=final.tolist(),
                    tracking=return_tracking,
                )
                print(f"Return movement verified: {args.joint}={final[joint_index]:.2f}", flush=True)

            held = hardware.hold()
            audit(args.log, "complete", held_state=held.tolist())
            print(json.dumps({"start": start.tolist(), "final": held.tolist()}, indent=2))
            return 0
        except Exception as exc:
            if hardware is not None:
                try:
                    held = hardware.hold()
                    audit(args.log, "stopped", error=str(exc), held_state=held.tolist())
                except Exception as hold_exc:
                    print(f"CRITICAL: hold failed: {hold_exc}", file=sys.stderr)
            raise
        finally:
            if hardware is not None:
                hardware.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"motion test refused/stopped: {exc}", file=sys.stderr)
        raise SystemExit(1)
