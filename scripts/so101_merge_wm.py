#!/usr/bin/env python3
"""Merge ALL datasets (VLA + exploratory) into one world-model corpus.

Sources: world-models, world-modelsv1..v19 (everything).
Output: shubham4413/so101_wm at <base>/so101_wm_merged.
"""
import json
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

BASE = Path("/media/shubhamnagar/One Touch/shubham4413_lerobot_data")
SETS = ["world-models"] + [f"world-modelsv{i}" for i in range(1, 20)]
AGGR_ROOT = BASE / "so101_wm_merged"

assert not AGGR_ROOT.exists(), f"{AGGR_ROOT} already exists -- remove it or pick a new name"
expected = 0
for s in SETS:
    p = BASE / s / "meta" / "info.json"
    assert p.exists(), f"missing dataset: {s}"
    expected += json.loads(p.read_text())["total_episodes"]
print(f"merging {len(SETS)} datasets, {expected} episodes expected")

aggregate_datasets(
    repo_ids=[f"shubham4413/{s}" for s in SETS],
    aggr_repo_id="shubham4413/so101_wm",
    roots=[BASE / s for s in SETS],
    aggr_root=AGGR_ROOT,
)

info = json.loads((AGGR_ROOT / "meta" / "info.json").read_text())
print("\n=== merged world-model corpus ===")
print(f"episodes: {info['total_episodes']} (expected {expected})")
print(f"frames:   {info['total_frames']} (~{info['total_frames']/30/60:.1f} min @ 30fps)")
print(f"tasks:    {info['total_tasks']} (expected 2: stacking + exploratory)")
