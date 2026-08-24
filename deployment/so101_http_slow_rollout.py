#!/usr/bin/env python3
"""Sequential hold-infer-move-verify client for an HPC-hosted VLA-JEPA policy."""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import os
import signal
import sys
import time
import tomllib
import uuid
from pathlib import Path
from threading import Event
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from so101_safety import (
    SafetyLimits,
    evaluate_tracking,
    plan_quintic_trajectory,
    sanitize_first_action,
    validate_action_chunk,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "deployment" / "so101_hardware.toml"
DEFAULT_TASK = "Grab the red hexagon on the right and place it on the red hexagon on the left."
ARM_PHRASE = "SAFE_SLOW_ROLLOUT"
LOCK_PATH = Path("/tmp/so101_http_slow_rollout.lock")
MOTOR_KEYS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class StopRequested(RuntimeError):
    pass


class Hardware:
    def __init__(self, config: dict, limits: SafetyLimits):
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.robots.so_follower.so_follower import SOFollower

        robot_cfg = config["robot"]
        relative_limit = dict(zip(limits.joint_names, limits.max_chunk_delta.tolist(), strict=True))
        self.robot = SOFollower(
            SOFollowerRobotConfig(
                port=robot_cfg["port"],
                id=robot_cfg["id"],
                cameras={},
                use_degrees=True,
                max_relative_target=relative_limit,
                # Holding the last bounded target is safer than making the arm go limp on a network error.
                disable_torque_on_disconnect=False,
            )
        )
        # Verify the cached calibration against the motor registers before configure() can write anything.
        self.robot.bus.connect(handshake=True)
        try:
            if not self.robot.bus.is_calibrated:
                raise RuntimeError("cached calibration does not match the connected arm")
            preflight_positions = self.robot.bus.sync_read("Present_Position")
            preflight_state = np.asarray(
                [preflight_positions[name] for name in MOTOR_KEYS], dtype=np.float64
            )
            validate_action_chunk(preflight_state[None, :], preflight_state, limits)
        finally:
            self.robot.bus.disconnect(disable_torque=False)

        self.robot.connect(calibrate=False)
        torque_state = self.robot.bus.sync_read("Torque_Enable")
        if not all(int(value) == 1 for value in torque_state.values()):
            self.robot.bus.enable_torque()
            torque_state = self.robot.bus.sync_read("Torque_Enable")
        if not all(int(value) == 1 for value in torque_state.values()):
            self.close()
            raise RuntimeError("one or more motors could not be torque-enabled")
        self.cameras: dict[str, cv2.VideoCapture] = {}
        try:
            for label in ("left", "fpv"):
                camera_cfg = config["cameras"][label]
                camera = cv2.VideoCapture(camera_cfg["device"], cv2.CAP_V4L2)
                camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_cfg["fourcc"]))
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg["width"])
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg["height"])
                camera.set(cv2.CAP_PROP_FPS, camera_cfg["fps"])
                for _ in range(12):
                    ok, frame = camera.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"camera warmup failed: {label}")
                self.cameras[label] = camera
        except Exception:
            self.close()
            raise

    def state(self) -> np.ndarray:
        positions = self.robot.bus.sync_read("Present_Position")
        return np.asarray([positions[name] for name in MOTOR_KEYS], dtype=np.float64)

    def observation(self, task: str) -> dict:
        observed_at_unix_ms = time.time_ns() / 1_000_000
        state = self.state()
        images = {}
        for label, camera in self.cameras.items():
            ok, frame = camera.read()
            if not ok or frame is None or frame.shape[:2] != (480, 640):
                raise RuntimeError(f"invalid live frame: {label}")
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError(f"JPEG encoding failed: {label}")
            images[label] = base64.b64encode(encoded).decode("ascii")
        return {
            "request_id": str(uuid.uuid4()),
            "observed_at_unix_ms": observed_at_unix_ms,
            "state": state.tolist(),
            "task": task,
            "left_jpeg": images["left"],
            "fpv_jpeg": images["fpv"],
        }

    def command(self, positions: np.ndarray) -> np.ndarray:
        action = {f"{name}.pos": float(value) for name, value in zip(MOTOR_KEYS, positions, strict=True)}
        sent = self.robot.send_action(action)
        return np.asarray([sent[f"{name}.pos"] for name in MOTOR_KEYS], dtype=np.float64)

    def hold(self) -> np.ndarray:
        current = self.state()
        self.command(current)
        return current

    def close(self) -> None:
        for camera in getattr(self, "cameras", {}).values():
            camera.release()
        robot = getattr(self, "robot", None)
        if robot is not None and robot.is_connected:
            robot.disconnect()


def require_local_tunnel(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("--url must be an HTTP loopback address reached through the SSH tunnel")


def audit(log_path: Path, event: str, **fields: object) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"unix_ms": time.time_ns() / 1_000_000, "event": event, **fields}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def request_action(
    session: requests.Session, url: str, payload: dict, timeout_s: float, max_age_ms: float
) -> tuple[dict, float]:
    started = time.monotonic()
    response = session.post(f"{url.rstrip('/')}/predict", json=payload, timeout=(3.0, timeout_s))
    round_trip_ms = (time.monotonic() - started) * 1_000
    response.raise_for_status()
    result = response.json()
    if result.get("request_id") != payload["request_id"]:
        raise RuntimeError("response request_id does not match the current observation")
    if round_trip_ms > max_age_ms:
        raise RuntimeError(f"response is stale ({round_trip_ms:.0f} ms > {max_age_ms:.0f} ms)")
    return result, round_trip_ms


def execute_trajectory(hardware: Hardware, trajectory: np.ndarray, hz: float, stop: Event) -> None:
    period = 1.0 / hz
    deadline = time.monotonic()
    for target in trajectory:
        if stop.is_set():
            raise StopRequested("stop requested")
        sent = hardware.command(target)
        if not np.allclose(sent, target, atol=1e-3, rtol=0):
            raise RuntimeError("LeRobot clipped a command; rollout stopped")
        deadline += period
        remaining = deadline - time.monotonic()
        if remaining > 0:
            stop.wait(remaining)
        elif remaining < -period:
            raise RuntimeError("local motor control loop missed its timing deadline")


def execute_and_verify(
    hardware: Hardware,
    start: np.ndarray,
    target: np.ndarray,
    trajectory: np.ndarray,
    limits: SafetyLimits,
    stop: Event,
) -> tuple[np.ndarray, dict]:
    """Execute, dwell on the final target, and require stable encoder-confirmed tracking."""
    execute_trajectory(hardware, trajectory, limits.control_hz, stop)
    period = 1.0 / limits.control_hz
    deadline = time.monotonic() + limits.settle_timeout_s
    stable_samples = 0
    measured = hardware.state()
    tracking_ok, report = evaluate_tracking(start, target, measured, limits)
    if not report["intended_joints"]:
        raise RuntimeError("model target contains no movement above the configured visible threshold")

    while time.monotonic() < deadline:
        if stop.is_set():
            raise StopRequested("stop requested while waiting for target tracking")
        sent = hardware.command(target)
        if not np.allclose(sent, target, atol=1e-3, rtol=0):
            raise RuntimeError("LeRobot clipped the final target during tracking")
        if stop.wait(period):
            raise StopRequested("stop requested while waiting for target tracking")
        measured = hardware.state()
        tracking_ok, report = evaluate_tracking(start, target, measured, limits)
        stable_samples = stable_samples + 1 if tracking_ok else 0
        if stable_samples >= limits.settle_required_samples:
            report["stable_samples"] = stable_samples
            return measured, report

    report["stable_samples"] = stable_samples
    raise RuntimeError(
        "robot did not visibly reach the model target before the settling timeout: "
        + json.dumps(report, separators=(",", ":"))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--token-env", default="SO101_INFERENCE_TOKEN")
    parser.add_argument("--cycles", type=int, default=1, help="Number of observe/infer/single-step cycles.")
    parser.add_argument("--arm", default="", help=f"Must equal {ARM_PHRASE!r} to permit motor commands.")
    parser.add_argument(
        "--log", type=Path, default=ROOT / "eval" / "out" / "slow_rollout.jsonl"
    )
    args = parser.parse_args()

    require_local_tunnel(args.url)
    if args.arm != ARM_PHRASE:
        raise RuntimeError(f"actuation is locked; pass --arm {ARM_PHRASE}")
    if not 1 <= args.cycles <= 10:
        raise ValueError("--cycles must be between 1 and 10")
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"authentication variable {args.token_env!r} is empty or unset")
    if not sys.stdin.isatty():
        raise RuntimeError("refusing actuation without an interactive terminal")

    print("Physical e-stop/power cut must be in reach. The arm will remain torque-enabled on exit.")
    confirmation = input(f"Type {ARM_PHRASE} again to run {args.cycles} bounded cycle(s): ")
    if confirmation != ARM_PHRASE:
        raise RuntimeError("arming confirmation did not match")

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    limits = SafetyLimits.from_config(config["safety"])
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"Authorization": f"Bearer {token}"})
    health = session.get(f"{args.url.rstrip('/')}/health", timeout=(3.0, args.timeout))
    health.raise_for_status()
    health_payload = health.json()
    if health_payload.get("status") != "ready" or health_payload.get("actuation") is not False:
        raise RuntimeError("unexpected inference-server health response")
    server_instance_id = health_payload.get("server_instance_id")
    if not isinstance(server_instance_id, str) or not server_instance_id:
        raise RuntimeError("inference-server health response has no instance identifier")

    stop = Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())

    hardware: Hardware | None = None
    with LOCK_PATH.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another slow-rollout client is already running") from exc
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        try:
            hardware = Hardware(config, limits)
            initial_state = hardware.state()
            validate_action_chunk(initial_state[None, :], initial_state, limits)
            audit(args.log, "armed", cycles=args.cycles, initial_state=initial_state.tolist())
            for cycle in range(args.cycles):
                if stop.is_set():
                    raise StopRequested("stop requested before observation")
                held_before_request = hardware.hold()
                payload = hardware.observation(args.task)
                observed_state = np.asarray(payload["state"], dtype=np.float64)
                # The motor controller holds this exact observation while the HTTP call blocks.
                hardware.command(observed_state)
                audit(
                    args.log,
                    "inference_wait_started",
                    cycle=cycle,
                    request_id=payload["request_id"],
                    held_state=held_before_request.tolist(),
                    observed_state=observed_state.tolist(),
                )
                result, round_trip_ms = request_action(
                    session, args.url, payload, args.timeout, limits.max_response_age_ms
                )
                current_instance = result.get("server_instance_id")
                if not isinstance(current_instance, str) or not current_instance:
                    raise RuntimeError("server response has no instance identifier")
                if current_instance != server_instance_id:
                    raise RuntimeError("inference server restarted during rollout")

                target, sanitization = sanitize_first_action(
                    result.get("action_chunk"), observed_state, limits
                )
                current_state = hardware.state()
                # The arm must still be close enough to the state on which the action was conditioned.
                if np.any(np.abs(current_state - observed_state) > limits.max_chunk_delta):
                    raise RuntimeError("robot moved too far while inference was running")
                # Re-apply the same displacement cap from the fresher pre-motion state.
                target, motion_sanitization = sanitize_first_action(
                    target[None, :], current_state, limits
                )
                trajectory = plan_quintic_trajectory(current_state, target, limits)
                if sanitization["clipped"] or sanitization["future_position_violation_count"]:
                    print(
                        "[safety] model output clipped/warned; executing only the bounded first action",
                        flush=True,
                    )
                audit(
                    args.log,
                    "prediction_accepted",
                    cycle=cycle,
                    request_id=payload["request_id"],
                    server_instance_id=server_instance_id,
                    round_trip_ms=round(round_trip_ms, 2),
                    inference_ms=result.get("inference_ms"),
                    observed_state=observed_state.tolist(),
                    original_target=sanitization["original_target"],
                    accepted_target=target.tolist(),
                    sanitization=sanitization,
                    motion_sanitization=motion_sanitization,
                    trajectory_samples=len(trajectory),
                )
                settled_state, tracking = execute_and_verify(
                    hardware, current_state, target, trajectory, limits, stop
                )
                held_state = hardware.hold()
                audit(
                    args.log,
                    "movement_complete",
                    cycle=cycle,
                    settled_state=settled_state.tolist(),
                    held_state=held_state.tolist(),
                    tracking=tracking,
                )
                print(
                    f"[cycle {cycle + 1}/{args.cycles}] target reached; holding before next request",
                    flush=True,
                )
            hardware.hold()
            audit(args.log, "complete", final_state=hardware.state().tolist())
            print("Bounded rollout completed; robot is holding its final position.")
            return 0
        except Exception as exc:
            if hardware is not None:
                try:
                    hardware.hold()
                except Exception as hold_exc:
                    print(f"CRITICAL: hold command failed: {hold_exc}", file=sys.stderr)
            audit(args.log, "stopped", error=type(exc).__name__, detail=str(exc))
            raise
        finally:
            if hardware is not None:
                hardware.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"slow rollout refused/stopped: {exc}", file=sys.stderr)
        raise SystemExit(1)
