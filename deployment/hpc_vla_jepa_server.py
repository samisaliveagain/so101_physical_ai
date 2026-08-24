#!/usr/bin/env python3
"""Minimal HTTP inference server for the SO-101 VLA-JEPA policy.

The server has no robot or camera access. It accepts a six-joint state, two JPEG
images, and a task instruction, then returns a postprocessed action chunk in the
dataset's physical joint units.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import io
import json
import math
import os
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image


RENAME_MAP = {
    "observation.images.left": "observation.images.exterior_1_left",
    "observation.images.fpv": "observation.images.exterior_2_left",
}
IMAGE_FIELDS = {
    "left_jpeg": "observation.images.left",
    "fpv_jpeg": "observation.images.fpv",
}
MAX_REQUEST_BYTES = 8 * 1024 * 1024


class ServerBusyError(RuntimeError):
    """Raised instead of queueing observations behind an active inference."""


def load_policy(policy_path: str, repo_id: str, dataset_root: str | None, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path
    cfg.device = device
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    policy = make_policy(cfg, ds_meta=metadata, rename_map=RENAME_MAP)
    policy.eval().to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def decode_image(encoded: str) -> torch.Tensor:
    raw = base64.b64decode(encoded, validate=True)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    if image.size != (640, 480):
        raise ValueError(f"expected a 640x480 image, got {image.size[0]}x{image.size[1]}")
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


class InferenceRuntime:
    def __init__(self, args: argparse.Namespace):
        self.device = args.device
        self.auth_token = os.environ.get(args.auth_token_env) if args.auth_token_env else None
        self.max_observation_age_ms = args.max_observation_age_ms
        self.server_instance_id = str(uuid.uuid4())
        self.policy, self.preprocessor, self.postprocessor = load_policy(
            args.policy_path, args.repo_id, args.dataset_root, args.device
        )
        self.lock = threading.Lock()

    @torch.inference_mode()
    def predict(self, request: dict) -> dict:
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("request_id must be a non-empty string of at most 128 characters")
        observed_at_unix_ms = request.get("observed_at_unix_ms")
        if not isinstance(observed_at_unix_ms, (int, float)) or not math.isfinite(observed_at_unix_ms):
            raise ValueError("observed_at_unix_ms must be a finite number")
        received_at_unix_ms = time.time_ns() / 1_000_000
        observation_age_ms = received_at_unix_ms - float(observed_at_unix_ms)
        if observation_age_ms < -2_000:
            raise ValueError("observation timestamp is more than 2 seconds in the future")
        if observation_age_ms > self.max_observation_age_ms:
            raise ValueError(
                f"observation is stale ({observation_age_ms:.0f} ms > "
                f"{self.max_observation_age_ms:.0f} ms)"
            )
        state = np.asarray(request.get("state"), dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError("state must contain exactly six finite joint positions")
        task = request.get("task")
        if not isinstance(task, str) or not task.strip() or len(task) > 500:
            raise ValueError("task must be a non-empty string of at most 500 characters")

        batch: dict[str, object] = {
            "observation.state": torch.from_numpy(state).unsqueeze(0),
            "task": [task.strip()],
        }
        for request_key, observation_key in IMAGE_FIELDS.items():
            encoded = request.get(request_key)
            if not isinstance(encoded, str):
                raise ValueError(f"missing base64 JPEG field: {request_key}")
            batch[observation_key] = decode_image(encoded)

        start = time.perf_counter()
        if not self.lock.acquire(blocking=False):
            raise ServerBusyError("inference already in progress; observation was not queued")
        try:
            self.policy.reset()
            self.preprocessor.reset()
            self.postprocessor.reset()
            processed = self.preprocessor(batch)
            normalized_chunk = self.policy.predict_action_chunk(processed)
            physical_steps = [
                self.postprocessor(normalized_chunk[:, step]).squeeze(0).float().cpu()
                for step in range(normalized_chunk.shape[1])
            ]
        finally:
            self.lock.release()
        chunk = torch.stack(physical_steps).numpy()
        if chunk.ndim != 2 or chunk.shape[1] != 6 or not np.isfinite(chunk).all():
            raise RuntimeError(f"policy returned invalid action chunk shape {chunk.shape}")
        return {
            "request_id": request_id,
            "server_instance_id": self.server_instance_id,
            "observed_at_unix_ms": observed_at_unix_ms,
            "received_at_unix_ms": round(received_at_unix_ms, 3),
            "responded_at_unix_ms": round(time.time_ns() / 1_000_000, 3),
            "action_chunk": chunk.tolist(),
            "action_horizon": int(chunk.shape[0]),
            "action_dim": int(chunk.shape[1]),
            "inference_ms": round((time.perf_counter() - start) * 1000.0, 2),
            "device": self.device,
            "actuation": False,
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "SO101VLAJEPA/1.0"

    @property
    def runtime(self) -> InferenceRuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    def send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def authorized(self) -> bool:
        token = self.runtime.auth_token
        supplied = self.headers.get("Authorization", "")
        return token is None or hmac.compare_digest(supplied, f"Bearer {token}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self.authorized():
            self.send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        self.send_json(
            HTTPStatus.OK,
            {
                "status": "ready",
                "device": self.runtime.device,
                "server_instance_id": self.runtime.server_instance_id,
                "actuation": False,
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self.authorized():
            self.send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            self.connection.settimeout(10.0)
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError(f"request body must be 1..{MAX_REQUEST_BYTES} bytes")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("request body must be a JSON object")
            result = self.runtime.predict(request)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except ServerBusyError as exc:
            self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
            return
        except Exception as exc:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": type(exc).__name__, "detail": str(exc)})
            return
        self.send_json(HTTPStatus.OK, result)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--repo-id", default="shubham4413/so101_wm")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument(
        "--auth-token-env",
        default=None,
        help="Optional environment-variable name containing a bearer token.",
    )
    parser.add_argument(
        "--max-observation-age-ms",
        type=float,
        default=10_000.0,
        help="Reject observations older than this at server receipt (default: 10000).",
    )
    parser.add_argument(
        "--allow-unauthenticated-nonloopback",
        action="store_true",
        help="Unsafe override: permit a non-loopback bind without bearer authentication.",
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is visible; start this through salloc+srun")
    if args.max_observation_age_ms <= 0:
        raise ValueError("--max-observation-age-ms must be positive")
    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    if (
        args.host not in loopback_hosts
        and not args.auth_token_env
        and not args.allow_unauthenticated_nonloopback
    ):
        raise RuntimeError(
            "refusing an unauthenticated non-loopback bind; set --auth-token-env or explicitly "
            "pass --allow-unauthenticated-nonloopback"
        )
    if args.auth_token_env and not os.environ.get(args.auth_token_env):
        raise RuntimeError(f"authentication environment variable {args.auth_token_env!r} is empty or unset")

    print("[server] loading policy; this can take several minutes", flush=True)
    runtime = InferenceRuntime(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.runtime = runtime  # type: ignore[attr-defined]
    print(f"[server] ready on http://{args.host}:{args.port} (actuation disabled)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
