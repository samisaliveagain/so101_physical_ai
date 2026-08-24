#!/usr/bin/env python3
"""Create deterministic, episode-level train/val/test splits for LeRobot data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-episodes", type=int, default=5)
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        split = json.loads(args.output.read_text())
        print(f"Reusing {args.output}: train={len(split['train'])}, val={len(split['val'])}, "
              f"test={len(split['test'])}")
        return

    info_path = args.dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"LeRobot metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    count = int(info["total_episodes"])
    holdout = args.val_episodes + args.test_episodes
    if count <= holdout:
        raise SystemExit(f"Need more than {holdout} episodes, found {count}")

    shuffled = list(range(count))
    random.Random(args.seed).shuffle(shuffled)
    test = sorted(shuffled[: args.test_episodes])
    val = sorted(shuffled[args.test_episodes : holdout])
    train = sorted(shuffled[holdout:])
    split = {
        "dataset_root": str(args.dataset_root.resolve()),
        "seed": args.seed,
        "total_episodes": count,
        "train": train,
        "val": val,
        "test": test,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2) + "\n")
    print(f"Wrote {args.output}: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    main()
