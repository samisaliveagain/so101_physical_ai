#!/usr/bin/env python3
"""Compare one Gazebo native link pose against the corresponding ROS TF."""

import argparse
import json
import math
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


def vector(data):
    return tuple(
        float(data.get(axis, 0.0) if isinstance(data, dict) else getattr(data, axis))
        for axis in ("x", "y", "z")
    )


def quaternion(data):
    return tuple(
        float(data.get(axis, 0.0) if isinstance(data, dict) else getattr(data, axis))
        for axis in ("x", "y", "z", "w")
    )


def multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate(q, v):
    rotated = multiply(multiply(q, (*v, 0.0)), (-q[0], -q[1], -q[2], q[3]))
    return rotated[:3]


def gazebo_world_pose(world_name, model_name, link_name):
    topic = f"/world/{world_name}/dynamic_pose/info"
    result = subprocess.run(
        ["gz", "topic", "-e", "-t", topic, "-n", "1", "--json-output"],
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
    )
    message = json.loads(result.stdout)
    poses = {entry["name"]: entry for entry in message["pose"]}
    if model_name not in poses or link_name not in poses:
        raise RuntimeError(f"Missing {model_name!r} or {link_name!r} on {topic}")

    model = poses[model_name]
    link = poses[link_name]
    model_position = vector(model.get("position", {}))
    model_orientation = quaternion(model.get("orientation", {}))
    link_position = vector(link.get("position", {}))
    link_orientation = quaternion(link.get("orientation", {}))
    offset = rotate(model_orientation, link_position)
    return (
        tuple(model_position[i] + offset[i] for i in range(3)),
        multiply(model_orientation, link_orientation),
    )


def ros_tf_pose(frame, link_name):
    rclpy.init()
    node = Node("validate_gazebo_tf_sync")
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    del listener
    deadline = time.monotonic() + 8.0
    transform = None
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            transform = buffer.lookup_transform(frame, link_name, rclpy.time.Time())
            break
        except Exception:
            pass
    node.destroy_node()
    rclpy.shutdown()
    if transform is None:
        raise RuntimeError(f"Timed out waiting for TF {frame} -> {link_name}")
    return vector(transform.transform.translation), quaternion(transform.transform.rotation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", default="so101_dataset_world")
    parser.add_argument("--model", default="so101")
    parser.add_argument("--frame", default="world")
    parser.add_argument("--link", default="gripper_link")
    parser.add_argument("--position-tolerance", type=float, default=1e-5)
    parser.add_argument("--angle-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    gz_position, gz_orientation = gazebo_world_pose(args.world, args.model, args.link)
    tf_position, tf_orientation = ros_tf_pose(args.frame, args.link)
    position_error = math.sqrt(
        sum((gz_position[i] - tf_position[i]) ** 2 for i in range(3))
    )
    dot = abs(sum(gz_orientation[i] * tf_orientation[i] for i in range(4)))
    angle_error = 2.0 * math.acos(max(-1.0, min(1.0, dot)))

    print(f"Gazebo {args.link}: position={gz_position}, quaternion={gz_orientation}")
    print(f"ROS TF  {args.link}: position={tf_position}, quaternion={tf_orientation}")
    print(
        f"sync errors: position={position_error:.3e} m, "
        f"orientation={angle_error:.3e} rad"
    )
    if position_error > args.position_tolerance or angle_error > args.angle_tolerance:
        print("Gazebo/TF synchronization: FAIL", file=sys.stderr)
        return 1
    print("Gazebo/TF synchronization: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
