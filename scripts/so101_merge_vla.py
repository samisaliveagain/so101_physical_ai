#!/usr/bin/env python3
"""Merge VLA datasets world-modelsv1..v11 into one unified dataset.

Output: shubham4413/so101_vla at <base>/so101_vla_merged (external drive).
Run AFTER visually verifying the camera-swap fix on v6-v11.
"""
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

BASE = Path("/media/shubhamnagar/One Touch/shubham4413_lerobot_data")
SETS = [f"world-modelsv{i}" for i in range(1, 12)]  # v1..v11, test episode excluded
AGGR_ROOT = BASE / "so101_vla_merged"

assert not AGGR_ROOT.exists(), f"{AGGR_ROOT} already exists -- remove it or pick a new name"
for s in SETS:
    assert (BASE / s / "meta" / "info.json").exists(), f"missing dataset: {s}"

aggregate_datasets(
    repo_ids=[f"shubham4413/{s}" for s in SETS],
    aggr_repo_id="shubham4413/so101_vla",
    roots=[BASE / s for s in SETS],
    aggr_root=AGGR_ROOT,
)

# sanity check the result
import json
info = json.loads((AGGR_ROOT / "meta" / "info.json").read_text())
print("\n=== merged dataset ===")
print(f"episodes: {info['total_episodes']} (expected 149)")
print(f"frames:   {info['total_frames']}")
print(f"tasks:    {info['total_tasks']}")
