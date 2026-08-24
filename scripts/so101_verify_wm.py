#!/usr/bin/env python3
"""Verify the merged WM corpus: decode frames from the unverified tail episodes
(v17=172-173, v18=174, v19=175-176) plus a start/middle sample."""
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path("/media/shubhamnagar/One Touch/shubham4413_lerobot_data/so101_wm_merged")
ds = LeRobotDataset("shubham4413/so101_wm", root=ROOT)
print(f"loaded: {ds.num_episodes} episodes, {ds.num_frames} frames")

eps = ds.meta.episodes
starts = list(eps["dataset_from_index"])
ends = list(eps["dataset_to_index"])

for ep in [0, 100, 172, 174, 176]:
    mid = (starts[ep] + ends[ep]) // 2
    item = ds[int(mid)]
    for cam in ["fpv", "left"]:
        img = item[f"observation.images.{cam}"]
        arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        out = Path.home() / f"wm_ep{ep}_{cam}.jpg"
        Image.fromarray(arr).save(out)
        print(f"episode {ep} {cam} -> {out.name}")
print("done")
