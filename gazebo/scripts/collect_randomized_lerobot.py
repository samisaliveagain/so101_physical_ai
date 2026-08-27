#!/usr/bin/env python3
"""Collect randomized Gazebo stacking rollouts in native LeRobot format."""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectoryPoint

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from approach_nut import Kinematics
from randomize_nuts import apply_layout, sample_layout

JOINT_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
CONTROLLER_JOINTS = [name.removesuffix(".pos") for name in JOINT_NAMES]
HOME_JOINTS = [-0.02758486, -0.66724474, 0.63985970, 0.46787783, 0.69129414, -0.000837]
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
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )

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

    def reset_robot(self, duration=6.0):
        if not self.trajectory_client.wait_for_server(timeout_sec=8.0):
            raise RuntimeError("arm trajectory action is unavailable during episode reset")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = CONTROLLER_JOINTS
        point = JointTrajectoryPoint()
        point.positions = HOME_JOINTS
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        goal.trajectory.points = [point]
        future = self.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("controller rejected the episode-reset trajectory")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if (
            wrapped is None
            or wrapped.status != GoalStatus.STATUS_SUCCEEDED
            or wrapped.result.error_code != 0
        ):
            raise RuntimeError("episode-reset trajectory failed")


def layout_xy(layout):
    return (
        float(layout["source"]["x"]),
        float(layout["source"]["y"]),
        float(layout["destination"]["x"]),
        float(layout["destination"]["y"]),
    )


def layouts_are_near(first, second, threshold):
    first_xy = layout_xy(first)
    second_xy = layout_xy(second)
    source_distance = math.dist(first_xy[:2], second_xy[:2])
    destination_distance = math.dist(first_xy[2:], second_xy[2:])
    return source_distance < threshold and destination_distance < threshold


def load_excluded_layouts(dataset_root):
    if dataset_root is None:
        return []
    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise RuntimeError(f"no LeRobot parquet files found under excluded dataset {dataset_root}")
    layouts = {}
    for data_file in data_files:
        table = pq.read_table(
            data_file,
            columns=["episode_index", "observation.environment_state"],
        )
        episode_indices = table["episode_index"].to_pylist()
        environment_states = table["observation.environment_state"].to_pylist()
        for episode_index, values in zip(episode_indices, environment_states, strict=True):
            if episode_index in layouts:
                continue
            layouts[episode_index] = {
                "source": {"x": values[0], "y": values[1]},
                "destination": {"x": values[4], "y": values[5]},
            }
    return list(layouts.values())


def sample_unique_layout(rng, kinematics, previous_layouts, threshold):
    for _ in range(2000):
        layout = sample_layout(rng, kinematics)
        if not any(layouts_are_near(layout, previous, threshold) for previous in previous_layouts):
            return layout
    raise RuntimeError("could not sample a sufficiently distinct IK-valid layout")


def gazebo_model_poses(world):
    result = subprocess.run(
        [
            "gz", "topic", "-e", "-t", f"/world/{world}/pose/info",
            "-n", "1", "--json-output",
        ],
        text=True,
        capture_output=True,
        timeout=10.0,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to read Gazebo model poses: {result.stderr.strip()}")
    message = json.loads(result.stdout)
    return {pose["name"]: pose for pose in message.get("pose", [])}


def stack_succeeded(layout, world):
    poses = gazebo_model_poses(world)
    if "red_hex_nut" not in poses:
        raise RuntimeError("red_hex_nut was missing from the Gazebo pose message")
    position = poses["red_hex_nut"].get("position", {})
    nut_x = float(position.get("x", 0.0))
    nut_y = float(position.get("y", 0.0))
    nut_z = float(position.get("z", 0.0))
    destination = layout["destination"]
    xy_error = math.dist((nut_x, nut_y), (destination["x"], destination["y"]))
    height_above_table = nut_z - destination["z"]
    return xy_error <= 0.035 and height_above_table >= 0.020, xy_error, height_above_table


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
    parser.add_argument(
        "--exclude-dataset",
        type=Path,
        action="append",
        default=[],
        help="reject layouts close to any episode in this existing LeRobot dataset",
    )
    parser.add_argument(
        "--min-layout-distance",
        type=float,
        default=0.012,
        help="metres; source and destination must not both be closer than this to a previous layout",
    )
    parser.add_argument("--keep-failed", action="store_true", help="save failed rollout episodes instead of discarding them")
    args = parser.parse_args()
    if args.episodes < 1 or args.output.exists():
        parser.error("episodes must be positive and output must not already exist")
    if args.fps != 30:
        parser.error("fps must be 30 to match the fixed Gazebo camera update rate")
    if args.min_layout_distance <= 0:
        parser.error("min-layout-distance must be positive")
    args.exclude_dataset = [path.expanduser().resolve() for path in args.exclude_dataset]

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
    previous_layouts = []
    for excluded_dataset in args.exclude_dataset:
        previous_layouts.extend(load_excluded_layouts(excluded_dataset))
    print(
        f"Loaded {len(previous_layouts)} excluded layouts; requiring "
        f"{args.min_layout_distance * 1000:.0f} mm source/destination pair separation"
    )
    rclpy.init()
    node = Recorder(dataset)
    try:
        wait_for_topics(node)
        saved = 0
        attempts = 0
        while saved < args.episodes:
            attempts += 1
            layout = sample_unique_layout(
                rng, kinematics, previous_layouts, args.min_layout_distance
            )
            node.reset_robot()
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
            node.last_stamp = None
            node.recording = True
            returncode = run_rollout(node, layout)
            node.recording = False
            success = False
            xy_error = float("inf")
            stack_height = float("-inf")
            if returncode == 0:
                success, xy_error, stack_height = stack_succeeded(layout, args.world)
            if success or args.keep_failed:
                dataset.save_episode()
                saved += 1
                previous_layouts.append(layout)
                print(
                    f"Saved episode {saved}/{args.episodes}: {node.frames} frames; "
                    f"success={success}; final_xy_error={xy_error * 1000:.1f} mm; "
                    f"stack_height={stack_height * 1000:.1f} mm; layout={layout}"
                )
            else:
                dataset.clear_episode_buffer()
                print(
                    f"Discarded failed rollout attempt {attempts}: returncode={returncode}, "
                    f"final_xy_error={xy_error * 1000:.1f} mm, "
                    f"stack_height={stack_height * 1000:.1f} mm",
                    file=sys.stderr,
                )
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
