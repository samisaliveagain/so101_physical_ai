#!/usr/bin/env python3
"""Evaluate the live VLA-JEPA HTTP pipeline without robot actuation."""

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
import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK = "Grab the red hexagon on the right and place it on the red hexagon on the left."
ACTION_MIN = np.array([-124.351646, -120.747253, -97.582420, -108.219780, -180.0, 0.0])
ACTION_MAX = np.array([112.307693, 105.362640, 97.494507, 108.835167, 180.0, 100.0])
FRAME_DELTA_Q99 = np.array([2.461538, 4.043957, 4.835167, 4.131866, 4.307693, 5.311355])
OFFLINE_OVERALL_MAE = 4.6983
CHUNK_BUDGET_MS = 7 / 30 * 1000
RANGE_TOLERANCE = 1e-4


def encode_jpeg(frame: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(encoded).decode("ascii")


def encode_file(path: Path) -> str:
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"could not read {path}")
    return encode_jpeg(frame)


class LiveObservationReader:
    """Persistent read-only serial and camera handles."""

    def __init__(self, config: dict):
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.robots.so_follower.so_follower import SOFollower

        robot_cfg = config["robot"]
        self.robot = SOFollower(
            SOFollowerRobotConfig(port=robot_cfg["port"], id=robot_cfg["id"], cameras={})
        )
        self.robot.bus.connect(handshake=True)
        self.cameras = {}
        for label, camera_cfg in config["cameras"].items():
            camera = cv2.VideoCapture(camera_cfg["device"], cv2.CAP_V4L2)
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_cfg["fourcc"]))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg["width"])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg["height"])
            camera.set(cv2.CAP_PROP_FPS, camera_cfg["fps"])
            for _ in range(12):
                ok, _ = camera.read()
                if not ok:
                    raise RuntimeError(f"camera warmup failed: {label}")
            self.cameras[label] = camera

    def read(self) -> dict:
        positions = self.robot.bus.sync_read("Present_Position")
        state = [float(positions[name]) for name in self.robot.bus.motors]
        images = {}
        for label, camera in self.cameras.items():
            ok, frame = camera.read()
            if not ok or frame is None or frame.shape[:2] != (480, 640):
                raise RuntimeError(f"invalid live frame: {label}")
            images[label] = encode_jpeg(frame)
        return {
            "state": state,
            "task": DEFAULT_TASK,
            "left_jpeg": images["left"],
            "fpv_jpeg": images["fpv"],
        }

    def close(self) -> None:
        for camera in self.cameras.values():
            camera.release()
        # False is intentional: disabling torque is a register write.
        self.robot.bus.disconnect(disable_torque=False)


def request_prediction(session: requests.Session, url: str, payload: dict, timeout: float) -> dict:
    payload = {
        **payload,
        "request_id": str(uuid.uuid4()),
        "observed_at_unix_ms": time.time_ns() / 1_000_000,
    }
    start = time.perf_counter()
    response = session.post(f"{url.rstrip('/')}/predict", json=payload, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    if result.get("request_id") != payload["request_id"]:
        raise RuntimeError("server response request_id does not match this observation")
    result["round_trip_ms"] = (time.perf_counter() - start) * 1000
    chunk = np.asarray(result["action_chunk"], dtype=float)
    result["_chunk"] = chunk
    return result


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "deployment/so101_hardware.toml")
    parser.add_argument("--heldout", type=Path, default=ROOT / "hardware/heldout_http/samples.json")
    parser.add_argument("--out", type=Path, default=ROOT / "eval/out/http_shadow_evaluation.json")
    parser.add_argument("--repeat-samples", type=int, default=10)
    parser.add_argument("--live-samples", type=int, default=30)
    parser.add_argument("--heldout-repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--token-env", default="SO101_INFERENCE_TOKEN")
    args = parser.parse_args()

    with args.hardware_config.open("rb") as handle:
        hardware = tomllib.load(handle)
    heldout = json.loads(args.heldout.read_text())
    session = requests.Session()
    token = os.environ.get(args.token_env)
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})
    reader = LiveObservationReader(hardware)
    all_results: list[dict] = []
    try:
        frozen_payload = reader.read()
        repeated = [
            request_prediction(session, args.url, frozen_payload, args.timeout)
            for _ in range(args.repeat_samples)
        ]
        live = []
        for _ in range(args.live_samples):
            payload = reader.read()
            result = request_prediction(session, args.url, payload, args.timeout)
            result["_state"] = payload["state"]
            live.append(result)
    finally:
        reader.close()

    heldout_results = []
    for sample in heldout:
        sample_dir = args.heldout.parent
        payload = {
            "state": sample["state"],
            "task": sample["task"],
            "left_jpeg": encode_file(sample_dir / sample["left_image"]),
            "fpv_jpeg": encode_file(sample_dir / sample["fpv_image"]),
        }
        predictions = [
            request_prediction(session, args.url, payload, args.timeout)
            for _ in range(args.heldout_repeats)
        ]
        heldout_results.append((sample, predictions))

    all_results = repeated + live + [r for _, results in heldout_results for r in results]
    chunks = [r["_chunk"] for r in all_results]
    all_actions = np.concatenate(chunks, axis=0)
    valid_shape = all(chunk.shape == (7, 6) for chunk in chunks)
    finite = all(np.isfinite(chunk).all() for chunk in chunks)
    within_demo_range = all(
        ((chunk >= ACTION_MIN - RANGE_TOLERANCE) & (chunk <= ACTION_MAX + RANGE_TOLERANCE)).all()
        for chunk in chunks
    )
    chunk_delta_ratios = [
        np.max(np.abs(np.diff(chunk, axis=0)) / FRAME_DELTA_Q99, axis=0) for chunk in chunks
    ]
    smooth_chunks = bool(np.max(chunk_delta_ratios) <= 1.0)

    frozen_state = np.asarray(frozen_payload["state"], dtype=float)
    repeated_first = np.stack([r["_chunk"][0] for r in repeated])
    first_delta_ratios = [
        np.abs(first - frozen_state) / FRAME_DELTA_Q99 for first in repeated_first
    ]
    first_delta_ratios.extend(
        np.abs(result["_chunk"][0] - np.asarray(result["_state"], dtype=float))
        / FRAME_DELTA_Q99
        for result in live
    )
    first_step_safe = bool(np.max(first_delta_ratios) <= 1.0)
    repeat_peak_to_peak = np.ptp(repeated_first, axis=0)
    repeatability_pass = bool(np.all(repeat_peak_to_peak <= 2 * FRAME_DELTA_Q99))

    heldout_errors = []
    heldout_gripper = []
    for sample, results in heldout_results:
        ground_truth = np.asarray(sample["action"], dtype=float)
        for result in results:
            prediction = result["_chunk"][0]
            heldout_errors.append(np.abs(prediction - ground_truth))
            heldout_gripper.append((prediction[5] > 0.5) == (ground_truth[5] > 0.5))
    heldout_errors_np = np.stack(heldout_errors)
    heldout_mae = float(heldout_errors_np.mean())
    heldout_threshold = OFFLINE_OVERALL_MAE * 1.5
    heldout_fidelity_pass = heldout_mae <= heldout_threshold
    gripper_accuracy = float(np.mean(heldout_gripper))
    gripper_pass = gripper_accuracy >= 0.8

    inference = [float(r["inference_ms"]) for r in all_results]
    round_trip = [float(r["round_trip_ms"]) for r in all_results]
    latency_p95 = percentile(inference, 95)
    round_trip_p95 = percentile(round_trip, 95)
    latency_pass = latency_p95 <= CHUNK_BUDGET_MS
    stale_response_pass = round_trip_p95 <= CHUNK_BUDGET_MS

    gates = {
        "http_shape_7x6": valid_shape,
        "finite_outputs": finite,
        "targets_within_demonstrated_range": within_demo_range,
        "within_chunk_q99_smoothness": smooth_chunks,
        "frozen_first_step_within_demo_q99": first_step_safe,
        "repeated_input_stability": repeatability_pass,
        "heldout_action_fidelity": heldout_fidelity_pass,
        "heldout_gripper_accuracy": gripper_pass,
        "server_latency_under_chunk_budget": latency_pass,
        "round_trip_under_chunk_budget": stale_response_pass,
        "actuation_disabled": True,
    }
    report = {
        "summary": {
            "passed": sum(gates.values()),
            "total": len(gates),
            "all_passed": all(gates.values()),
            "actuation_approved": all(gates.values()),
        },
        "gates": gates,
        "thresholds": {
            "action_min": ACTION_MIN.tolist(),
            "action_max": ACTION_MAX.tolist(),
            "frame_delta_q99": FRAME_DELTA_Q99.tolist(),
            "chunk_budget_ms": CHUNK_BUDGET_MS,
            "range_numerical_tolerance": RANGE_TOLERANCE,
            "heldout_mae_limit": heldout_threshold,
        },
        "measurements": {
            "requests": len(all_results),
            "repeat_requests": len(repeated),
            "live_requests": len(live),
            "heldout_requests": sum(len(results) for _, results in heldout_results),
            "inference_ms_p50": percentile(inference, 50),
            "inference_ms_p95": latency_p95,
            "inference_ms_max": max(inference),
            "round_trip_ms_p50": percentile(round_trip, 50),
            "round_trip_ms_p95": round_trip_p95,
            "round_trip_ms_max": max(round_trip),
            "repeat_first_action_peak_to_peak": repeat_peak_to_peak.tolist(),
            "max_live_first_step_q99_ratio": float(np.max(first_delta_ratios)),
            "max_chunk_delta_q99_ratio": float(np.max(chunk_delta_ratios)),
            "observed_action_min": all_actions.min(axis=0).tolist(),
            "observed_action_max": all_actions.max(axis=0).tolist(),
            "below_demonstrated_min_count": (all_actions < ACTION_MIN).sum(axis=0).tolist(),
            "above_demonstrated_max_count": (all_actions > ACTION_MAX).sum(axis=0).tolist(),
            "max_below_demonstrated_min": np.maximum(ACTION_MIN - all_actions.min(axis=0), 0).tolist(),
            "max_above_demonstrated_max": np.maximum(all_actions.max(axis=0) - ACTION_MAX, 0).tolist(),
            "heldout_overall_mae": heldout_mae,
            "heldout_per_joint_mae": heldout_errors_np.mean(axis=0).tolist(),
            "heldout_gripper_binary_accuracy": gripper_accuracy,
        },
        "actuation": False,
        "robot_commands_sent": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
