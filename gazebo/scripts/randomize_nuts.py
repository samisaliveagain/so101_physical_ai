#!/usr/bin/env python3
"""Sample reachable, non-overlapping SO-101 stacking-part poses."""

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from approach_nut import BASE_XYZ, Kinematics

TABLE_Z = 0.631
DEFAULT_SEED_JOINTS = [-0.02758485964, -0.66724473550, 0.63985970198, 0.46787783282, 0.69129414425]

# Reachable spawn regions established around the successful stacking layout.
# These are intentionally smaller than the table: every sampled layout must
# still pass the complete grasp/lift/transfer/place IK check below.
SOURCE_X_RANGE = (0.060, 0.150)
SOURCE_Y_RANGE = (0.075, 0.170)
DESTINATION_X_RANGE = (0.055, 0.155)
DESTINATION_Y_RANGE = (-0.005, 0.085)
MIN_PART_SEPARATION = 0.110

# The reproducible layout stored in so101_dataset.sdf comes from successful
# training episode 42.  Interactive evaluation jitters around this anchor so a
# new Gazebo launch is different without immediately testing a large
# out-of-distribution displacement.
REFERENCE_SOURCE = (0.13052899, 0.13765612, 2.89777589)
REFERENCE_DESTINATION = (0.07708498, 0.02392881, -1.06950760)


def radial(x, y):
    dx, dy = BASE_XYZ[0] - x, BASE_XYZ[1] - y
    length = math.hypot(dx, dy)
    return dx / length, dy / length


def ik_reachable(kinematics, source, destination):
    q = DEFAULT_SEED_JOINTS
    points = [
        (source[0], source[1], TABLE_Z + 0.080, radial(*source)),
        (source[0], source[1], TABLE_Z + 0.022, radial(*source)),
        (source[0], source[1], TABLE_Z + 0.120, radial(*source)),
        (destination[0], destination[1], TABLE_Z + 0.120, radial(*destination)),
        (destination[0], destination[1], TABLE_Z + 0.052, radial(*destination)),
    ]
    try:
        for x, y, z, direction in points:
            q, _, _ = kinematics.ik((x, y, z), q, direction, restarts=2)
    except RuntimeError:
        return False
    return True


def sample_layout(rng, kinematics):
    # Sample a broader part of the demonstrated workspace. Full-range yaw is
    # useful even for the nearly symmetric nut because it changes the exact
    # finger-to-rim contact geometry in Gazebo.
    for _ in range(400):
        source = (rng.uniform(*SOURCE_X_RANGE), rng.uniform(*SOURCE_Y_RANGE))
        destination = (rng.uniform(*DESTINATION_X_RANGE), rng.uniform(*DESTINATION_Y_RANGE))
        if math.dist(source, destination) < MIN_PART_SEPARATION:
            continue
        if ik_reachable(kinematics, source, destination):
            return {
                "source": {"x": source[0], "y": source[1], "z": TABLE_Z, "yaw": rng.uniform(-math.pi, math.pi)},
                "destination": {
                    "x": destination[0], "y": destination[1], "z": TABLE_Z, "yaw": rng.uniform(-math.pi, math.pi)
                },
            }
    raise RuntimeError("could not sample an IK-valid layout after 400 attempts")


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def sample_near_reference(rng, kinematics, xy_jitter=0.015, yaw_jitter=math.radians(15.0)):
    """Jitter the known-good evaluation layout while retaining IK validity."""
    if xy_jitter < 0.0 or yaw_jitter < 0.0:
        raise ValueError("jitter values must be non-negative")

    for _ in range(400):
        source = (
            REFERENCE_SOURCE[0] + rng.uniform(-xy_jitter, xy_jitter),
            REFERENCE_SOURCE[1] + rng.uniform(-xy_jitter, xy_jitter),
        )
        destination = (
            REFERENCE_DESTINATION[0] + rng.uniform(-xy_jitter, xy_jitter),
            REFERENCE_DESTINATION[1] + rng.uniform(-xy_jitter, xy_jitter),
        )
        if math.dist(source, destination) < MIN_PART_SEPARATION:
            continue
        if ik_reachable(kinematics, source, destination):
            return {
                "source": {
                    "x": source[0],
                    "y": source[1],
                    "z": TABLE_Z,
                    "yaw": wrap_angle(REFERENCE_SOURCE[2] + rng.uniform(-yaw_jitter, yaw_jitter)),
                },
                "destination": {
                    "x": destination[0],
                    "y": destination[1],
                    "z": TABLE_Z,
                    "yaw": wrap_angle(REFERENCE_DESTINATION[2] + rng.uniform(-yaw_jitter, yaw_jitter)),
                },
            }
    raise RuntimeError("could not sample an IK-valid near-reference layout after 400 attempts")


def wait_for_pose_service(world="so101_dataset_world", timeout=20.0):
    service = f"/world/{world}/set_pose"
    deadline = time.monotonic() + timeout
    available = []
    while time.monotonic() < deadline:
        result = subprocess.run(["gz", "service", "-l"], text=True, capture_output=True, check=False)
        available = [name for name in result.stdout.splitlines() if name.endswith("/set_pose")]
        if service in result.stdout.splitlines():
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"Gazebo service {service} was not discovered after {timeout:.0f} seconds; "
        f"available set_pose services: {available or 'none'}")


def set_model_pose(model, pose, world="so101_dataset_world"):
    half = pose["yaw"] / 2.0
    request = (
        f'name: "{model}", position: {{x: {pose["x"]}, y: {pose["y"]}, z: {pose["z"]}}}, '
        f'orientation: {{z: {math.sin(half)}, w: {math.cos(half)}}}'
    )
    last_output = ""
    for attempt in range(1, 6):
        result = subprocess.run(
            ["gz", "service", "-s", f"/world/{world}/set_pose", "--reqtype", "gz.msgs.Pose",
             "--reptype", "gz.msgs.Boolean", "--timeout", "10000", "--req", request],
            text=True, capture_output=True, check=False,
        )
        last_output = (result.stdout + result.stderr).strip()
        if result.returncode == 0 and "data: true" in result.stdout.lower():
            return
        if attempt < 5:
            time.sleep(0.75)
    raise RuntimeError(f"failed to move {model} after 5 attempts: {last_output}")


def apply_layout(layout, world="so101_dataset_world"):
    wait_for_pose_service(world)
    set_model_pose("red_hex_nut", layout["source"], world)
    set_model_pose("red_hex_cap", layout["destination"], world)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--world", default="so101_dataset_world")
    parser.add_argument("--dry-run", action="store_true", help="sample and print without changing Gazebo")
    parser.add_argument(
        "--near-reference", action="store_true",
        help="sample around the known-good evaluation layout instead of the full collection workspace",
    )
    parser.add_argument(
        "--xy-jitter", type=float, default=0.015, metavar="METRES",
        help="maximum per-axis offset in near-reference mode (default: 0.015)",
    )
    parser.add_argument(
        "--yaw-jitter-deg", type=float, default=15.0, metavar="DEGREES",
        help="maximum yaw offset in near-reference mode (default: 15)",
    )
    parser.add_argument("--urdf", type=Path, default=SCRIPT_DIR.parent / "models/so101_dark_blue/so101_dark_blue.urdf")
    args = parser.parse_args()
    if args.xy_jitter < 0.0 or args.yaw_jitter_deg < 0.0:
        parser.error("jitter values must be non-negative")

    rng = random.Random(args.seed)
    kinematics = Kinematics(args.urdf)
    if args.near_reference:
        layout = sample_near_reference(
            rng,
            kinematics,
            xy_jitter=args.xy_jitter,
            yaw_jitter=math.radians(args.yaw_jitter_deg),
        )
    else:
        layout = sample_layout(rng, kinematics)
    if not args.dry_run:
        apply_layout(layout, args.world)
    print(json.dumps(layout, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
