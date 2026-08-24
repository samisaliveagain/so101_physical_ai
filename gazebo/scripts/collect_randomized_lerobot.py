#!/usr/bin/env python3
"""Collect randomized Gazebo stacking rollouts in native LeRobot format."""

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rclpy.node import Node
from sensor_msgs.msg import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from approach_nut import Kinematics
from randomize_nuts import apply_layout, sample_layout

JOINT_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
TASK = "Pick up the red nut and stack it on the other red part."


def image_to_rgb(message):
    channels = 4 if message.encoding in ("rgba8", "bgra8") else 3
    row = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = row[:, : message.width * channels].reshape(message.height, message.width, channels)
    if message.encoding in ("bgr8", "bgra8"):
        image = image[:, :, [2, 1, 0] + ([3] if channels == 4 else [])]
    if channels == 4:
        image = image[:, :, :3]
    if message.encoding not in ("rgb8", "bgr8", "rgba8", "bgra8"):
        raise RuntimeError(f"unsupported camera encoding {message.encoding!r}")
    return np.ascontiguousarray(image)


class Recorder(Node):
    def __init__(self, dataset):
        super().__init__("so101_lerobot_recorder")
        self.dataset = dataset
        self.controller = None
        self.left = None
        self.fpv = None
        self.recording = False
        self.layout_vector = None
        self.last_stamp = None
        self.frames = 0
        self.create_subscription(JointTrajectoryControllerState, "/arm_controller/controller_state", self._controller, 20)
        self.create_subscription(Image, "/so101/camera/left/image", self._left, 10)
        self.create_subscription(Image, "/so101/camera/fpv/image", self._fpv, 10)

    def _controller(self, message):
        self.controller = message

    def _fpv(self, message):
        self.fpv = image_to_rgb(message)

    def _left(self, message):
        self.left = image_to_rgb(message)
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        if (not self.recording or self.layout_vector is None or stamp == self.last_stamp or
                self.controller is None or self.fpv is None):
            return
        if len(self.controller.feedback.positions) != 6 or len(self.controller.reference.positions) != 6:
            return
        self.last_stamp = stamp
        self.dataset.add_frame({
            "observation.state": np.asarray(self.controller.feedback.positions, dtype=np.float32),
            "observation.environment_state": self.layout_vector,
            "action": np.asarray(self.controller.reference.positions, dtype=np.float32),
            "observation.images.left": self.left.copy(),
            "observation.images.fpv": self.fpv.copy(),
            "task": TASK,
        })
        self.frames += 1


def wait_for_topics(node, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and (node.controller is None or node.left is None or node.fpv is None):
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.controller is None or node.left is None or node.fpv is None:
        raise RuntimeError("controller state and both camera topics were not ready; start run_sim_rviz.sh first")


def run_rollout(node, layout, timeout=90.0):
    command = [
        str(SCRIPT_DIR / "approach_nut.sh"), "--x", str(layout["source"]["x"]), "--y", str(layout["source"]["y"]),
        "--z", str(layout["source"]["z"]), "--place-x", str(layout["destination"]["x"]),
        "--place-y", str(layout["destination"]["y"]),
    ]
    process = subprocess.Popen(command, text=True)
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
    if process.poll() is None:
        process.terminate()
        raise RuntimeError("rollout exceeded 90 seconds")
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--repo-id", default="local/so101_gazebo_randomized_stack")
    parser.add_argument("--output", type=Path, default=Path("data/so101_gazebo_randomized_stack"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--world", default="so101_dataset_world")
    parser.add_argument("--keep-failed", action="store_true", help="save failed rollout episodes instead of discarding them")
    args = parser.parse_args()
    if args.episodes < 1 or args.output.exists():
        parser.error("episodes must be positive and output must not already exist")
    if args.fps != 30:
        parser.error("fps must be 30 to match the fixed Gazebo camera update rate")

    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "observation.environment_state": {
            "dtype": "float32", "shape": (8,),
            "names": ["source_spawn_x", "source_spawn_y", "source_spawn_z", "source_spawn_yaw",
                      "destination_spawn_x", "destination_spawn_y", "destination_spawn_z", "destination_spawn_yaw"],
        },
        "observation.images.left": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
        "observation.images.fpv": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
    }
    dataset = LeRobotDataset.create(
        args.repo_id, args.fps, features, root=args.output.resolve(), robot_type="so101_gazebo",
        use_videos=True, image_writer_threads=4, vcodec="libsvtav1",
    )
    rng = random.Random(args.seed)
    kinematics = Kinematics(SCRIPT_DIR.parent / "models/so101_dark_blue/so101_dark_blue.urdf")
    rclpy.init()
    node = Recorder(dataset)
    try:
        wait_for_topics(node)
        saved = 0
        attempts = 0
        while saved < args.episodes:
            attempts += 1
            layout = sample_layout(rng, kinematics)
            apply_layout(layout, args.world)
            source, destination = layout["source"], layout["destination"]
            node.layout_vector = np.asarray(
                [source["x"], source["y"], source["z"], source["yaw"], destination["x"],
                 destination["y"], destination["z"], destination["yaw"]], dtype=np.float32)
            # Let the dynamic nut settle after teleportation.
            settle_until = time.monotonic() + 1.0
            while time.monotonic() < settle_until:
                rclpy.spin_once(node, timeout_sec=0.02)
            node.frames = 0
            node.recording = True
            returncode = run_rollout(node, layout)
            node.recording = False
            if returncode == 0 or args.keep_failed:
                dataset.save_episode()
                saved += 1
                print(f"Saved episode {saved}/{args.episodes}: {node.frames} frames; layout={layout}")
            else:
                dataset.clear_episode_buffer()
                print(f"Discarded failed rollout attempt {attempts}", file=sys.stderr)
            if attempts >= args.episodes * 4 and saved < args.episodes:
                raise RuntimeError("too many failed randomized rollouts")
        dataset.finalize()
        print(f"LeRobot dataset finalized at {args.output.resolve()}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
