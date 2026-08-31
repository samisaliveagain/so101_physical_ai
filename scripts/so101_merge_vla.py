#!/usr/bin/env python3
"""Merge the stacking demonstration shards into one LeRobot dataset."""

import argparse
import json
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

SETS = [f"world-modelsv{i}" for i in range(1, 12)]  # v1..v11, test episode excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, required=True, help="directory containing source shards"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repo-id", default="local/so101_vla")
    args = parser.parse_args()

    output = args.output or args.base / "so101_vla_merged"
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    missing = [name for name in SETS if not (args.base / name / "meta/info.json").is_file()]
    if missing:
        raise FileNotFoundError(f"missing dataset shards: {', '.join(missing)}")

    aggregate_datasets(
        repo_ids=[f"local/{name}" for name in SETS],
        aggr_repo_id=args.repo_id,
        roots=[args.base / name for name in SETS],
        aggr_root=output,
    )
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    print(f"episodes: {info['total_episodes']}")
    print(f"frames:   {info['total_frames']}")
    print(f"tasks:    {info['total_tasks']}")


if __name__ == "__main__":
    main()
