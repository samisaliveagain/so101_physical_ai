#!/usr/bin/env python3
"""Merge stacking and exploratory shards into one LeRobot corpus."""

import argparse
import json
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

SETS = ["world-models"] + [f"world-modelsv{i}" for i in range(1, 20)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, required=True, help="directory containing source shards"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repo-id", default="local/so101_wm")
    args = parser.parse_args()

    output = args.output or args.base / "so101_wm_merged"
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    metadata = [args.base / name / "meta/info.json" for name in SETS]
    missing = [str(path) for path in metadata if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing dataset metadata: {', '.join(missing)}")
    expected = sum(
        json.loads(path.read_text(encoding="utf-8"))["total_episodes"]
        for path in metadata
    )

    aggregate_datasets(
        repo_ids=[f"local/{name}" for name in SETS],
        aggr_repo_id=args.repo_id,
        roots=[args.base / name for name in SETS],
        aggr_root=output,
    )
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    print(f"episodes: {info['total_episodes']} (expected {expected})")
    print(f"frames:   {info['total_frames']}")
    print(f"tasks:    {info['total_tasks']}")


if __name__ == "__main__":
    main()
