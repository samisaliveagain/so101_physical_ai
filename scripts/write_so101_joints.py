#!/usr/bin/env python3
"""Move one SO-101 joint, hold it, and return using direct Goal_Position writes."""

from __future__ import annotations

import argparse
import time

from so101_bus import load_hardware_config, setup_bus


def move_to_pose(bus, desired: dict[str, float], duration: float) -> dict[str, float]:
    start = bus.sync_read("Present_Position")
    started = time.monotonic()
    while True:
        elapsed = time.monotonic() - started
        alpha = min(elapsed / duration, 1.0)
        command = {joint: (1 - alpha) * start[joint] + alpha * goal for joint, goal in desired.items()}
        bus.sync_write("Goal_Position", command, normalize=True)
        if alpha >= 1.0:
            break
        time.sleep(0.02)
    return bus.sync_read("Present_Position")


def hold_position(bus, desired: dict[str, float], duration: float) -> dict[str, float]:
    started = time.monotonic()
    while time.monotonic() - started < duration:
        bus.sync_write("Goal_Position", desired, normalize=True)
        time.sleep(0.02)
    return bus.sync_read("Present_Position")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", default="wrist_roll")
    parser.add_argument("--delta", type=float, default=3.0)
    parser.add_argument("--move-time", type=float, default=2.0)
    parser.add_argument("--hold-time", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true", help="Required to enable physical movement.")
    args = parser.parse_args()
    if not args.execute:
        raise RuntimeError("physical movement is locked; pass --execute")
    if not 0 < args.move_time <= 10 or not 0 <= args.hold_time <= 10:
        raise ValueError("move-time must be in (0,10] and hold-time in [0,10]")

    config = load_hardware_config()
    safety = config["safety"]
    if args.joint not in safety["joint_names"]:
        raise ValueError(f"unknown joint {args.joint!r}")
    index = safety["joint_names"].index(args.joint)
    if abs(args.delta) > safety["max_model_delta"][index]:
        raise ValueError(f"delta exceeds the configured limit for {args.joint}")

    bus = setup_bus(config)
    try:
        bus.enable_torque()
        start = bus.sync_read("Present_Position")
        target = dict(start)
        target[args.joint] += args.delta
        if not safety["joint_min"][index] <= target[args.joint] <= safety["joint_max"][index]:
            raise ValueError("target lies outside the configured joint range")

        print("start:", start)
        print("target:", target)
        reached = move_to_pose(bus, target, args.move_time)
        print("reached:", reached)
        held = hold_position(bus, target, args.hold_time)
        print("held:", held)
        returned = move_to_pose(bus, start, args.move_time)
        print("returned:", returned)
        hold_position(bus, start, 0.5)
    except Exception:
        current = bus.sync_read("Present_Position")
        bus.sync_write("Goal_Position", current, normalize=True)
        raise
    finally:
        # Leave torque enabled at the final position so the arm does not collapse.
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
