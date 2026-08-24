#!/usr/bin/env python3
"""Run LeRobot training with checkpoint pointers that also work on exFAT.

LeRobot normally creates ``checkpoints/last`` as a symbolic link.  Filesystems
commonly used by removable drives, including exFAT, do not support symbolic
links.  The checkpoint itself is already complete when that operation occurs,
so replace only the pointer update with a portable fallback marker.
"""

from __future__ import annotations

import errno
import sys
from pathlib import Path

from lerobot.scripts import lerobot_train


def update_last_checkpoint_portable(checkpoint_dir: Path) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints_dir = checkpoint_dir.parent
    last_checkpoint = checkpoints_dir / "last"
    relative_target = checkpoint_dir.relative_to(checkpoints_dir)

    if last_checkpoint.is_symlink():
        last_checkpoint.unlink()

    try:
        last_checkpoint.symlink_to(relative_target, target_is_directory=True)
    except OSError as exc:
        unsupported = {
            errno.EPERM,
            errno.EACCES,
            errno.ENOSYS,
            errno.EOPNOTSUPP,
        }
        if exc.errno not in unsupported:
            raise

        marker = checkpoints_dir / "last_checkpoint.txt"
        marker.write_text(f"{relative_target.as_posix()}\n", encoding="utf-8")
        print(
            f"WARNING: {checkpoints_dir} does not support symbolic links; "
            f"recorded the latest checkpoint in {marker} instead.",
            file=sys.stderr,
        )


lerobot_train.update_last_checkpoint = update_last_checkpoint_portable


if __name__ == "__main__":
    lerobot_train.main()
