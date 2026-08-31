#!/usr/bin/env python3
"""Decode representative frames from a merged LeRobot dataset."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--repo-id", default="local/so101_vla")
    parser.add_argument("--output", type=Path, default=Path("verification_frames"))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("=== video file sizes ===")
    for camera in ["fpv", "left"]:
        video_dir = root / "videos" / f"observation.images.{camera}"
        for path in sorted(video_dir.rglob("*.mp4")):
            relative = path.relative_to(root / "videos")
            print(f"  {camera}: {relative}  {path.stat().st_size / 1e6:.1f} MB")

    dataset = LeRobotDataset(args.repo_id, root=root)
    print(f"\nloaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    episodes = dataset.meta.episodes
    starts = list(episodes["dataset_from_index"])
    ends = list(episodes["dataset_to_index"])

    for episode in sorted({0, dataset.num_episodes // 2, dataset.num_episodes - 1}):
        middle = (starts[episode] + ends[episode]) // 2
        item = dataset[int(middle)]
        for camera in ["fpv", "left"]:
            image = item[f"observation.images.{camera}"]
            array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            path = output / f"merged_ep{episode}_{camera}.jpg"
            Image.fromarray(array).save(path)
            print(f"episode {episode} {camera}: frame decoded -> {path.name}")

    print("\nAll representative frames decoded successfully.")


if __name__ == "__main__":
    main()
