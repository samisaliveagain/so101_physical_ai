#!/usr/bin/env python3
"""Read calibrated SO-101 joint positions through the LeRobot motor-bus API."""

from __future__ import annotations

import argparse
import time

from so101_bus import load_hardware_config, setup_bus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", action="store_true", help="Print positions at 50 Hz until Ctrl-C.")
    parser.add_argument("--free", action="store_true", help="Disable torque so the arm can be moved by hand.")
    args = parser.parse_args()

    bus = setup_bus(load_hardware_config())
    try:
        if args.free:
            print("Torque disabled: support the arm before moving it by hand.")
            bus.disable_torque()
        while True:
            print(bus.sync_read("Present_Position"))
            if not args.continuous:
                break
            time.sleep(0.02)
    finally:
        # Preserve the current torque state; do not unexpectedly make the arm limp.
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
