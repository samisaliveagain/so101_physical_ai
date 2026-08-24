#!/usr/bin/env python3
"""Verify the merged VLA dataset by decoding frames through lerobot's loader.

Checks episodes 0 (from v1), ~74 (middle), and 148 (last, from v11):
saves one frame per camera per episode and prints video file sizes.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path("/media/shubhamnagar/One Touch/shubham4413_lerobot_data/so101_vla_merged")

print("=== video file sizes ===")
for cam in ["fpv", "left"]:
    for f in sorted((ROOT / "videos" / f"observation.images.{cam}").rglob("*.mp4")):
        print(f"  {cam}: {f.relative_to(ROOT / 'videos')}  {f.stat().st_size / 1e6:.1f} MB")

ds = LeRobotDataset("shubham4413/so101_vla", root=ROOT)
print(f"\nloaded: {ds.num_episodes} episodes, {ds.num_frames} frames")

# global frame index of the middle frame of an episode
eps = ds.meta.episodes
starts = list(eps["dataset_from_index"])
ends = list(eps["dataset_to_index"])

for ep in [0, 74, 148]:
    mid = (starts[ep] + ends[ep]) // 2
    item = ds[int(mid)]
    for cam in ["fpv", "left"]:
        img = item[f"observation.images.{cam}"]
        arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        out = Path.home() / f"merged_ep{ep}_{cam}.jpg"
        Image.fromarray(arr).save(out)
        print(f"episode {ep} {cam}: frame decoded -> {out.name}")

print("\nAll decodes OK -- merge integrity confirmed at start/middle/end.")
