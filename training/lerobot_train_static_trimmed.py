#!/usr/bin/env python3
"""Train LeRobot while reducing long stationary runs in the sample index.

The source LeRobot dataset and its videos remain untouched.  The wrapper only
changes which observation frames may be selected as training examples; action
chunks are still resolved from the original, contiguous episode timeline.
This is important for ACT because physically deleting frames would change the
meaning of a 30 Hz action chunk.
"""

from __future__ import annotations

import errno
import json
import os
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from lerobot.datasets.factory import make_dataset as make_dataset_untrimmed
from lerobot.scripts import lerobot_train


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _nonnegative_float_env(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value, got {value}")
    return value


def stationary_trim_indices(
    actions: np.ndarray,
    states: np.ndarray,
    episode_indices: np.ndarray,
    threshold: float,
    max_static_frames: int,
    final_static_frames: int,
) -> tuple[np.ndarray, dict]:
    """Return sample indices after capping stationary runs episode-by-episode.

    For an internal stationary run, retain its final frames so the ACT target
    chunk includes the next motion.  For the last stationary run in an episode,
    retain its first few frames to preserve the demonstrated stop without
    oversampling the completed pose.
    """
    actions = np.asarray(actions, dtype=np.float32)
    states = np.asarray(states, dtype=np.float32)
    episode_indices = np.asarray(episode_indices, dtype=np.int64)
    if actions.ndim != 2 or states.shape != actions.shape:
        raise ValueError(f"expected matching 2-D action/state arrays, got {actions.shape} and {states.shape}")
    if len(episode_indices) != len(actions):
        raise ValueError("episode index length does not match action/state arrays")
    if max_static_frames < 1 or not 1 <= final_static_frames <= max_static_frames:
        raise ValueError("static-frame caps must satisfy 1 <= final <= maximum")

    keep = np.ones(len(actions), dtype=bool)
    capped_runs = 0
    longest_run = 0
    episodes = np.unique(episode_indices)
    for episode in episodes:
        indices = np.flatnonzero(episode_indices == episode)
        if not len(indices):
            continue
        if len(indices) > 1 and not np.all(np.diff(indices) == 1):
            raise ValueError(f"episode {episode} is not contiguous in the selected dataset")

        local_actions = actions[indices]
        local_states = states[indices]
        if len(indices) == 1:
            continue
        static_transition = (
            np.max(np.abs(np.diff(local_actions, axis=0)), axis=1) <= threshold
        ) & (
            np.max(np.abs(np.diff(local_states, axis=0)), axis=1) <= threshold
        )

        start = 0
        while start < len(indices):
            end = start
            while end < len(indices) - 1 and static_transition[end]:
                end += 1
            run_length = end - start + 1
            longest_run = max(longest_run, run_length)
            if run_length > max_static_frames:
                keep[indices[start : end + 1]] = False
                if end == len(indices) - 1:
                    keep[indices[start : start + final_static_frames]] = True
                else:
                    keep[indices[end - max_static_frames + 1 : end + 1]] = True
                capped_runs += 1
            start = end + 1

    selected = np.flatnonzero(keep)
    report = {
        "original_frames": int(len(actions)),
        "retained_frames": int(len(selected)),
        "removed_frames": int(len(actions) - len(selected)),
        "retained_percent": round(100.0 * len(selected) / max(len(actions), 1), 3),
        "episodes": int(len(episodes)),
        "threshold_rad": threshold,
        "max_static_frames": max_static_frames,
        "final_static_frames": final_static_frames,
        "capped_static_runs": capped_runs,
        "longest_static_run_frames": longest_run,
    }
    return selected, report


class StaticTrimmedDataset(Dataset):
    """Index-filtering view that delegates decoding and ACT windows to LeRobot."""

    def __init__(self, dataset, selected_indices: np.ndarray):
        self.dataset = dataset
        self.selected_indices = np.asarray(selected_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.selected_indices)

    def __getitem__(self, index: int) -> dict:
        return self.dataset[int(self.selected_indices[index])]

    @property
    def num_frames(self) -> int:
        return len(self.selected_indices)

    def __getattr__(self, name):
        # Avoid recursion if a multiprocessing worker inspects the object while
        # its attributes are being restored.
        if name == "dataset":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "dataset"), name)


def make_static_trimmed_dataset(cfg):
    dataset = make_dataset_untrimmed(cfg)
    raw = dataset.select_columns(["action", "observation.state", "episode_index"])
    actions = np.asarray(raw["action"], dtype=np.float32)
    states = np.asarray(raw["observation.state"], dtype=np.float32)
    episode_indices = np.asarray(raw["episode_index"], dtype=np.int64)

    threshold = _nonnegative_float_env("SO101_STATIC_THRESHOLD_RAD", 1e-4)
    max_static_frames = _positive_int_env("SO101_MAX_STATIC_FRAMES", 15)
    final_static_frames = _positive_int_env("SO101_FINAL_STATIC_FRAMES", 5)
    selected, report = stationary_trim_indices(
        actions,
        states,
        episode_indices,
        threshold,
        max_static_frames,
        final_static_frames,
    )
    print("Static-frame sampling report: " + json.dumps(report, sort_keys=True))

    report_path = os.environ.get("SO101_STATIC_TRIM_REPORT")
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return StaticTrimmedDataset(dataset, selected)


def update_last_checkpoint_portable(checkpoint_dir: Path) -> None:
    """Use a text marker when the external-drive filesystem lacks symlinks."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints_dir = checkpoint_dir.parent
    last_checkpoint = checkpoints_dir / "last"
    relative_target = checkpoint_dir.relative_to(checkpoints_dir)
    if last_checkpoint.is_symlink():
        last_checkpoint.unlink()
    try:
        last_checkpoint.symlink_to(relative_target, target_is_directory=True)
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EACCES, errno.ENOSYS, errno.EOPNOTSUPP}:
            raise
        marker = checkpoints_dir / "last_checkpoint.txt"
        marker.write_text(f"{relative_target.as_posix()}\n", encoding="utf-8")
        print(f"WARNING: wrote portable checkpoint pointer to {marker}", file=sys.stderr)


lerobot_train.make_dataset = make_static_trimmed_dataset
lerobot_train.update_last_checkpoint = update_last_checkpoint_portable


if __name__ == "__main__":
    lerobot_train.main()
