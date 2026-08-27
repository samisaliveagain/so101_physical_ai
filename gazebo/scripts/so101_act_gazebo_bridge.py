#!/usr/bin/env python3
"""Run a LeRobot ACT checkpoint against the SO-101 Gazebo ROS interfaces."""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
import torch
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_URDF = SCRIPT_DIR.parent / "models/so101_dark_blue/so101_dark_blue.urdf"
DEFAULT_CHECKPOINT = Path(
    "/media/shubhamnagar/One Touch/so101_training/act/"
    "act_so101_gazebo_randomized_stack_20260824_005251_20260824_182755/"
    "checkpoints/100000/pretrained_model"
)
JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
IMAGE_SHAPE = (480, 640, 3)


def image_to_rgb(message: Image) -> np.ndarray:
    """Convert a ROS image to the exact uint8 HWC RGB representation used for training."""
    encodings = {"rgb8": (3, False), "bgr8": (3, True), "rgba8": (4, False), "bgra8": (4, True)}
    if message.encoding not in encodings:
        raise RuntimeError(f"unsupported camera encoding {message.encoding!r}")
    channels, is_bgr = encodings[message.encoding]
    row = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = row[:, : message.width * channels].reshape(message.height, message.width, channels)
    if is_bgr:
        image = image[:, :, [2, 1, 0] + ([3] if channels == 4 else [])]
    image = image[:, :, :3]
    if image.shape != IMAGE_SHAPE:
        raise RuntimeError(f"camera image is {image.shape}, expected {IMAGE_SHAPE}")
    return np.ascontiguousarray(image)


def load_joint_limits(urdf: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(urdf).getroot()
    limits = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name in JOINTS and limit is not None:
            limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
    missing = [name for name in JOINTS if name not in limits]
    if missing:
        raise RuntimeError(f"URDF has no position limits for: {', '.join(missing)}")
    lower = np.asarray([limits[name][0] for name in JOINTS], dtype=np.float32)
    upper = np.asarray([limits[name][1] for name in JOINTS], dtype=np.float32)
    return lower, upper


def ordered_positions(message: JointTrajectoryControllerState) -> np.ndarray:
    if len(message.feedback.positions) != len(message.joint_names):
        raise RuntimeError("controller feedback joint-name and position lengths differ")
    values = dict(zip(message.joint_names, message.feedback.positions, strict=True))
    missing = [name for name in JOINTS if name not in values]
    if missing:
        raise RuntimeError(f"controller feedback is missing joints: {', '.join(missing)}")
    return np.asarray([values[name] for name in JOINTS], dtype=np.float32)


def load_act(checkpoint: Path, requested_device: str):
    # Importing ACT registers its config type with LeRobot's config dispatcher.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    required = [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"checkpoint is incomplete; missing: {', '.join(missing)}")

    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested, but PyTorch cannot access the NVIDIA GPU")

    config = PreTrainedConfig.from_pretrained(
        checkpoint, device=device, local_files_only=True
    )
    expected_inputs = {
        "observation.state",
        "observation.images.left",
        "observation.images.fpv",
    }
    if set(config.input_features) != expected_inputs:
        raise RuntimeError(
            f"checkpoint input keys are {sorted(config.input_features)}, expected {sorted(expected_inputs)}"
        )
    if tuple(config.output_features["action"].shape) != (6,):
        raise RuntimeError("checkpoint action does not contain the six SO-101 joints")

    policy = ACTPolicy.from_pretrained(
        checkpoint, config=config, local_files_only=True
    ).eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor, torch.device(device)


class ActGazeboBridge(Node):
    def __init__(self, args, policy, preprocessor, postprocessor, device):
        super().__init__("so101_act_gazebo_bridge")
        self.args = args
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.lower, self.upper = load_joint_limits(args.urdf)

        self.state = None
        self.state_time = None
        self.state_buffer = deque(maxlen=args.state_buffer_size)
        self.left_buffer = deque(maxlen=args.camera_buffer_size)
        self.fpv_buffer = deque(maxlen=args.camera_buffer_size)
        self.left_sequence = 0
        self.last_processed_left_sequence = -1
        self.ready_error = None
        self.running = False
        self.done = False
        self.steps = 0
        self.start_time = None
        self.last_sync_time = None

        self.create_subscription(
            JointTrajectoryControllerState,
            args.state_topic,
            self._controller_callback,
            20,
        )
        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=args.camera_buffer_size,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Image, args.left_topic, self._left_callback, camera_qos)
        self.create_subscription(Image, args.fpv_topic, self._fpv_callback, camera_qos)
        self.command_publisher = self.create_publisher(JointTrajectory, args.command_topic, 10)
        self.raw_publisher = self.create_publisher(JointState, "/so101/act/predicted_action", 10)
        self.safe_publisher = self.create_publisher(JointState, "/so101/act/commanded_action", 10)

    def _controller_callback(self, message):
        try:
            self.state = ordered_positions(message)
            self.state_time = time.monotonic()
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            self.state_buffer.append((stamp, self.state_time, self.state.copy()))
        except RuntimeError as exc:
            self.ready_error = str(exc)

    def _left_callback(self, message):
        try:
            image = image_to_rgb(message)
            receipt_time = time.monotonic()
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            self.left_sequence += 1
            self.left_buffer.append((stamp, receipt_time, self.left_sequence, image))
        except RuntimeError as exc:
            self.ready_error = str(exc)

    def _fpv_callback(self, message):
        try:
            image = image_to_rgb(message)
            receipt_time = time.monotonic()
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            self.fpv_buffer.append((stamp, receipt_time, image))
        except RuntimeError as exc:
            self.ready_error = str(exc)

    def topics_ready(self) -> bool:
        return bool(self.state_buffer and self.left_buffer and self.fpv_buffer)

    def _synchronized_observation(self):
        """Return the newest close left/FPV/state timestamp match, or None."""
        best_camera_skew = float("inf")
        best_state_skew = float("inf")
        for left_stamp, left_time, left_sequence, left_image in reversed(self.left_buffer):
            if left_sequence <= self.last_processed_left_sequence:
                continue
            fpv_stamp, fpv_time, fpv_image = min(
                self.fpv_buffer, key=lambda frame: abs(frame[0] - left_stamp)
            )
            camera_skew = abs(fpv_stamp - left_stamp)
            best_camera_skew = min(best_camera_skew, camera_skew)
            if camera_skew > self.args.max_camera_skew:
                continue
            observation_stamp = 0.5 * (left_stamp + fpv_stamp)
            state_stamp, state_time, state = min(
                self.state_buffer, key=lambda frame: abs(frame[0] - observation_stamp)
            )
            state_skew = abs(state_stamp - observation_stamp)
            best_state_skew = min(best_state_skew, state_skew)
            if state_skew > self.args.max_state_skew:
                continue
            return {
                "left": left_image,
                "fpv": fpv_image,
                "state": state,
                "left_sequence": left_sequence,
                "camera_skew": camera_skew,
                "state_skew": state_skew,
                "receipt_times": (left_time, fpv_time, state_time),
            }, best_camera_skew, best_state_skew
        return None, best_camera_skew, best_state_skew

    def start(self):
        self.policy.reset()
        self.start_time = time.monotonic()
        self.last_sync_time = self.start_time
        self.running = True
        mode = "EXECUTE" if self.args.execute else "DRY RUN"
        self.get_logger().info(
            f"ACT bridge started in {mode} mode at {self.args.rate:.1f} Hz on {self.device}; "
            f"replanning every {self.args.action_horizon} actions; "
            f"controller output: {self.args.command_topic}; "
            f"raw predictions: /so101/act/predicted_action"
        )

    def _publish_joint_state(self, publisher, positions):
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = JOINTS
        message.position = [float(value) for value in positions]
        publisher.publish(message)

    def _bounded_action(self, raw_action):
        limited = np.clip(raw_action, self.lower, self.upper)
        per_step = np.asarray(
            [self.args.max_arm_velocity / self.args.rate] * 5
            + [self.args.max_gripper_velocity / self.args.rate],
            dtype=np.float32,
        )
        limited = np.clip(limited, self.state - per_step, self.state + per_step)
        return np.clip(limited, self.lower, self.upper)

    def _publish_trajectory(self, action):
        trajectory = JointTrajectory()
        # A zero header stamp tells joint_trajectory_controller to start now.
        # Gazebo's controller uses simulation time, while this standalone node
        # otherwise uses wall time; mixing those clocks makes a trajectory look
        # billions of seconds early or late.
        trajectory.joint_names = JOINTS
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in action]
        nanoseconds = int(self.args.command_time * 1_000_000_000)
        point.time_from_start = Duration(
            sec=nanoseconds // 1_000_000_000,
            nanosec=nanoseconds % 1_000_000_000,
        )
        trajectory.points = [point]
        self.command_publisher.publish(trajectory)

    def _control_tick(self):
        if not self.running or self.done:
            return
        now = time.monotonic()
        if self.args.duration and now - self.start_time >= self.args.duration:
            self.done = True
            return
        if not self.topics_ready():
            return
        synchronized, best_camera_skew, best_state_skew = self._synchronized_observation()
        if synchronized is None:
            unsynchronized_for = now - self.last_sync_time
            if unsynchronized_for >= self.args.sync_warning_after:
                camera_detail = (
                    f"{best_camera_skew:.3f}s"
                    if best_camera_skew < float("inf")
                    else "unavailable"
                )
                state_detail = (
                    f"{best_state_skew:.3f}s"
                    if best_state_skew < float("inf")
                    else "unavailable"
                )
                self.get_logger().warning(
                    f"no synchronized camera/state triplet for {unsynchronized_for:.1f}s; "
                    f"closest camera skew={camera_detail}, state skew={state_detail}",
                    throttle_duration_sec=2.0,
                )
            return
        self.last_sync_time = now
        ages = [now - receipt_time for receipt_time in synchronized["receipt_times"]]
        if max(ages) > self.args.stale_timeout:
            self.get_logger().warning(
                f"waiting for fresh synchronized inputs; matched data age is {max(ages):.3f}s",
                throttle_duration_sec=2.0,
            )
            return

        from lerobot.utils.control_utils import predict_action

        observation = {
            "observation.state": synchronized["state"].copy(),
            "observation.images.left": synchronized["left"].copy(),
            "observation.images.fpv": synchronized["fpv"].copy(),
        }
        try:
            action_tensor = predict_action(
                observation,
                self.policy,
                self.device,
                self.preprocessor,
                self.postprocessor,
                use_amp=self.args.amp,
                robot_type="so101_gazebo",
            )
            raw_action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
        except Exception as exc:
            self.get_logger().error(f"ACT inference failed: {exc}")
            self.done = True
            return
        if raw_action.shape != (6,) or not np.all(np.isfinite(raw_action)):
            self.get_logger().error(f"invalid ACT action: shape={raw_action.shape}, values={raw_action}")
            self.done = True
            return

        safe_action = self._bounded_action(raw_action)
        self._publish_joint_state(self.raw_publisher, raw_action)
        self._publish_joint_state(self.safe_publisher, safe_action)
        if self.args.execute:
            self._publish_trajectory(safe_action)

        self.last_processed_left_sequence = synchronized["left_sequence"]
        self.steps += 1
        if self.steps == 1 or self.steps % self.args.log_every == 0:
            difference = float(np.max(np.abs(raw_action - safe_action)))
            self.get_logger().info(
                f"step={self.steps}, max safety clamp={difference:.4f} rad, "
                f"camera skew={synchronized['camera_skew']*1000:.1f}ms, "
                f"state skew={synchronized['state_skew']*1000:.1f}ms, "
                f"command={[round(float(value), 3) for value in safe_action]}"
            )
        if self.args.max_steps and self.steps >= self.args.max_steps:
            self.done = True
        if self.args.duration and now - self.start_time >= self.args.duration:
            self.done = True


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("SO101_ACT_CHECKPOINT", DEFAULT_CHECKPOINT)),
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action="store_true", help="use CUDA automatic mixed precision")
    parser.add_argument("--execute", action="store_true", help="send bounded predictions to Gazebo")
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=50,
        help="actions to execute from each 100-action ACT prediction before replanning",
    )
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=90.0, help="seconds; zero runs until Ctrl+C")
    parser.add_argument("--max-steps", type=int, default=0, help="zero disables the step limit")
    parser.add_argument("--command-time", type=float, default=0.10)
    parser.add_argument("--max-arm-velocity", type=float, default=1.0, help="rad/s safety clamp")
    parser.add_argument("--max-gripper-velocity", type=float, default=0.8, help="rad/s safety clamp")
    parser.add_argument("--stale-timeout", type=float, default=0.5)
    parser.add_argument("--max-camera-skew", type=float, default=0.05)
    parser.add_argument("--max-state-skew", type=float, default=0.05)
    parser.add_argument(
        "--sync-warning-after",
        type=float,
        default=1.0,
        help="seconds without a valid timestamp triplet before warning",
    )
    parser.add_argument("--camera-buffer-size", type=int, default=30)
    parser.add_argument("--state-buffer-size", type=int, default=100)
    parser.add_argument(
        "--callbacks-per-step",
        type=int,
        default=12,
        help="maximum queued ROS callbacks drained before each ACT prediction",
    )
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--state-topic", default="/arm_controller/controller_state")
    parser.add_argument("--left-topic", default="/so101/camera/left/image")
    parser.add_argument("--fpv-topic", default="/so101/camera/fpv/image")
    parser.add_argument("--command-topic", default="/arm_controller/joint_trajectory")
    args = parser.parse_args()
    if args.rate <= 0 or args.command_time <= 0:
        parser.error("rate and command-time must be positive")
    if args.max_arm_velocity <= 0 or args.max_gripper_velocity <= 0:
        parser.error("velocity limits must be positive")
    if args.max_camera_skew <= 0 or args.max_state_skew <= 0 or args.sync_warning_after <= 0:
        parser.error("camera/state skew limits and sync-warning-after must be positive")
    if args.camera_buffer_size < 2 or args.state_buffer_size < 2:
        parser.error("camera/state buffer sizes must be at least 2")
    if (
        args.duration < 0
        or args.max_steps < 0
        or args.log_every < 1
        or args.action_horizon < 1
        or args.callbacks_per_step < 3
    ):
        parser.error(
            "duration/max-steps must be nonnegative; log-every/action-horizon must be positive; "
            "callbacks-per-step must be at least 3"
        )
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.urdf = args.urdf.expanduser().resolve()
    return args


def main():
    args = parse_args()
    policy, preprocessor, postprocessor, device = load_act(args.checkpoint, args.device)
    if args.action_horizon > policy.config.chunk_size:
        raise RuntimeError(
            f"action horizon {args.action_horizon} exceeds checkpoint chunk size {policy.config.chunk_size}"
        )
    # This changes only how many queued predictions are executed. The trained
    # checkpoint still predicts its original 100-action chunk, but a fresh
    # observation is used after this shorter horizon is consumed.
    policy.config.n_action_steps = args.action_horizon
    policy.reset()
    print(f"Loaded ACT checkpoint directly from {args.checkpoint} on {device}", flush=True)

    rclpy.init()
    node = ActGazeboBridge(args, policy, preprocessor, postprocessor, device)
    try:
        deadline = time.monotonic() + args.ready_timeout
        while rclpy.ok() and not node.topics_ready() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.ready_error:
                raise RuntimeError(node.ready_error)
        if not node.topics_ready():
            raise RuntimeError(
                "controller state and both cameras were not received; start run_sim_rviz.sh first"
            )
        if args.execute:
            discovery_deadline = time.monotonic() + 5.0
            while (
                rclpy.ok()
                and node.command_publisher.get_subscription_count() == 0
                and time.monotonic() < discovery_deadline
            ):
                rclpy.spin_once(node, timeout_sec=0.1)
            if node.command_publisher.get_subscription_count() == 0:
                raise RuntimeError(f"controller is not subscribed to {args.command_topic}")
        node.start()
        while rclpy.ok() and not node.done:
            # ACT inference blocks this process while Gazebo keeps publishing.
            # Drain the resulting camera/state callbacks before predicting again.
            # Keeping inference outside a ROS timer prevents an overdue timer from
            # starving sensor callbacks indefinitely.
            rclpy.spin_once(node, timeout_sec=0.01)
            for _ in range(args.callbacks_per_step - 1):
                rclpy.spin_once(node, timeout_sec=0.0)
            node._control_tick()
        node.get_logger().info(f"ACT bridge stopped after {node.steps} actions")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, ValueError, ET.ParseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
