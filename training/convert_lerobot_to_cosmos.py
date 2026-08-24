#!/usr/bin/env python3
"""Convert an SO-101 LeRobot v3 dataset to Cosmos action-conditioned data.

Cosmos derives its 7-D conditions from consecutive end-effector poses. This converter
uses the simulation URDF for FK, writes a compact 5 Hz H.264 video per episode, and
stores [x,y,z,roll,pitch,yaw] plus normalized gripper closure in each annotation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
CHAIN = JOINTS + ["gripper_frame_joint"]
BASE_XYZ = (0.38, 0.22, 0.632)
BASE_RPY = (0.0, 0.0, math.pi)
GRIPPER_OPEN = 1.745
GRIPPER_CLOSED = -0.174


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, xyz[0]],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, xyz[1]],
                     [-sp, cp*sr, cp*cr, xyz[2]], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_to_rpy(rot: np.ndarray) -> list[float]:
    pitch = math.atan2(-rot[2, 0], math.hypot(rot[0, 0], rot[1, 0]))
    if abs(math.cos(pitch)) > 1e-7:
        roll = math.atan2(rot[2, 1], rot[2, 2])
        yaw = math.atan2(rot[1, 0], rot[0, 0])
    else:
        roll = math.atan2(-rot[1, 2], rot[1, 1])
        yaw = 0.0
    return [roll, pitch, yaw]


class SO101FK:
    def __init__(self, urdf: Path):
        joints = {j.attrib["name"]: j for j in ET.parse(urdf).getroot().findall("joint")}
        self.origins: list[np.ndarray] = []
        for name in CHAIN:
            origin = joints[name].find("origin")
            xyz = tuple(float(v) for v in origin.attrib.get("xyz", "0 0 0").split())
            rpy = tuple(float(v) for v in origin.attrib.get("rpy", "0 0 0").split())
            self.origins.append(transform(xyz, rpy))

    def pose(self, q: np.ndarray) -> list[float]:
        pose = transform(BASE_XYZ, BASE_RPY)
        for index, origin in enumerate(self.origins):
            pose = matmul(pose, origin)
            if index < len(JOINTS):
                pose = matmul(pose, transform(rpy=(0.0, 0.0, float(q[index]))))
        return [*pose[:3, 3].tolist(), *rotation_to_rpy(pose[:3, :3])]


class VideoWriter:
    def __init__(self, path: Path, fps: int, width: int, height: int):
        self.path = path
        self.tmp = path.with_suffix(".partial.mp4")
        self.container = av.open(str(self.tmp), mode="w", format="mp4")
        try:
            self.stream = self.container.add_stream("libx264", rate=fps)
        except av.error.FFmpegError:
            self.stream = self.container.add_stream("mpeg4", rate=fps)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"

    def append(self, image: torch.Tensor) -> None:
        array = (image.clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy())
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.tmp.replace(self.path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--camera-key", default="observation.images.left")
    parser.add_argument("--source-fps", type=int, default=30)
    parser.add_argument("--target-fps", type=int, default=5)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--urdf", type=Path, default=repo_root / "gazebo/models/so101_dark_blue/so101_dark_blue.urdf")
    parser.add_argument("--max-episodes", type=int, default=None, help="For a small conversion smoke test")
    args = parser.parse_args()

    if args.source_fps % args.target_fps:
        raise SystemExit("source-fps must be divisible by target-fps")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"Output is not empty: {args.output_root}. Choose a new directory.")
    split = json.loads(args.split_file.read_text())
    wanted = {ep: name for name in ("train", "val", "test") for ep in split[name]}
    selected = sorted(wanted)
    if args.max_episodes is not None:
        selected = selected[: args.max_episodes]

    for name in ("train", "val", "test"):
        (args.output_root / "annotation" / name).mkdir(parents=True, exist_ok=True)
        (args.output_root / "videos" / name).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_DATASETS_CACHE", str(args.output_root / ".hf_datasets_cache"))
    ds = LeRobotDataset("local/so101_gazebo_randomized_stack", root=args.dataset_root)
    fk = SO101FK(args.urdf)
    stride = args.source_fps // args.target_fps
    manifest = {"source": str(args.dataset_root), "camera_key": args.camera_key,
                "source_fps": args.source_fps, "target_fps": args.target_fps,
                "resolution": [args.height, args.width], "episodes": {}}

    for number, ep_index in enumerate(selected, 1):
        row = ds.meta.episodes[int(ep_index)]
        start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        frame_indices = list(range(start, stop, stride))
        if len(frame_indices) < 13:
            print(f"Skipping episode {ep_index}: only {len(frame_indices)} frames at {args.target_fps} Hz")
            continue
        subset = wanted[ep_index]
        video_rel = Path("videos") / subset / f"episode_{ep_index:06d}.mp4"
        video_abs = args.output_root / video_rel
        writer = VideoWriter(video_abs, args.target_fps, args.width, args.height)
        poses, grippers = [], []
        try:
            for frame_index in frame_indices:
                item = ds[frame_index]
                image = item[args.camera_key]
                image = F.interpolate(image.unsqueeze(0), size=(args.height, args.width),
                                      mode="bilinear", align_corners=False)[0]
                writer.append(image)
                q = item["observation.state"].detach().cpu().numpy()
                poses.append(fk.pose(q[:5]))
                closure = float(np.clip((GRIPPER_OPEN - q[5]) / (GRIPPER_OPEN - GRIPPER_CLOSED), 0.0, 1.0))
                grippers.append(closure)
        finally:
            writer.close()

        annotation = {
            "episode_id": f"so101_{ep_index:06d}",
            "state": poses,
            "continuous_gripper_state": grippers,
            "videos": [{"video_path": video_rel.as_posix()}],
        }
        ann_path = args.output_root / "annotation" / subset / f"episode_{ep_index:06d}.json"
        ann_path.write_text(json.dumps(annotation, separators=(",", ":")) + "\n")
        manifest["episodes"][str(ep_index)] = {"split": subset, "frames": len(frame_indices),
                                                "annotation": str(ann_path.relative_to(args.output_root))}
        print(f"[{number}/{len(selected)}] episode {ep_index}: {subset}, {len(frame_indices)} frames")

    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Cosmos dataset written to {args.output_root}")


if __name__ == "__main__":
    main()
