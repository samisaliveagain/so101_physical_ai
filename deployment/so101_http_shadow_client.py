#!/usr/bin/env python3
"""Capture one live SO-101 observation and request an HTTP prediction.

This program intentionally contains no robot actuation path. The serial bus is
used only to ping motors and read Present_Position.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import tomllib
import uuid
from pathlib import Path

import cv2
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "deployment" / "so101_hardware.toml"
DEFAULT_TASK = "Grab the red hexagon on the right and place it on the red hexagon on the left."


def capture_jpeg(config: dict) -> str:
    camera = cv2.VideoCapture(config["device"], cv2.CAP_V4L2)
    try:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config["fourcc"]))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
        camera.set(cv2.CAP_PROP_FPS, config["fps"])
        frame = None
        for _ in range(12):
            ok, frame = camera.read()
            if not ok:
                frame = None
                break
        if frame is None:
            raise RuntimeError(f"failed to capture {config['device']}")
        height, width = frame.shape[:2]
        if (width, height) != (640, 480):
            raise RuntimeError(f"unexpected camera resolution {width}x{height}")
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return base64.b64encode(encoded).decode("ascii")
    finally:
        camera.release()


def read_state(config: dict) -> list[float]:
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    robot = SOFollower(SOFollowerRobotConfig(port=config["port"], id=config["id"], cameras={}))
    robot.bus.connect(handshake=True)
    try:
        positions = robot.bus.sync_read("Present_Position")
        return [float(positions[name]) for name in robot.bus.motors]
    finally:
        robot.bus.disconnect(disable_torque=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--token", default=None)
    parser.add_argument("--token-env", default="SO101_INFERENCE_TOKEN")
    args = parser.parse_args()

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    print("[shadow] reading joint state; no register writes or goal commands", flush=True)
    state = read_state(config["robot"])
    print("[shadow] capturing left and fpv frames", flush=True)
    payload = {
        "request_id": str(uuid.uuid4()),
        "observed_at_unix_ms": time.time_ns() / 1_000_000,
        "state": state,
        "task": args.task,
        "left_jpeg": capture_jpeg(config["cameras"]["left"]),
        "fpv_jpeg": capture_jpeg(config["cameras"]["fpv"]),
    }
    token = args.token or os.environ.get(args.token_env)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    start = time.perf_counter()
    response = requests.post(f"{args.url.rstrip('/')}/predict", json=payload, headers=headers, timeout=args.timeout)
    response.raise_for_status()
    result = response.json()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    chunk = result["action_chunk"]
    if result.get("request_id") != payload["request_id"]:
        raise RuntimeError("server response request_id does not match this observation")
    first = chunk[0]
    delta = [action - position for action, position in zip(first, state, strict=True)]
    print(json.dumps({
        "state": state,
        "first_action": first,
        "first_action_delta": delta,
        "action_horizon": result["action_horizon"],
        "server_inference_ms": result["inference_ms"],
        "http_round_trip_ms": round(elapsed_ms, 2),
        "actuation": False,
    }, indent=2))
    print("[shadow] prediction received; no action was sent to the robot")


if __name__ == "__main__":
    main()
