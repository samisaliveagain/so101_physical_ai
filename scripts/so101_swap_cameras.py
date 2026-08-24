#!/usr/bin/env python3
"""Fix datasets whose fpv/left camera streams were recorded swapped.

Usage:
  python so101_swap_cameras.py inspect <dataset_root> [...]   # dump first frame of each cam to jpg
  python so101_swap_cameras.py swap    <dataset_root> [...]   # swap fpv <-> left everywhere

The swap covers: video directories, meta/stats.json entries,
meta/episodes parquet per-camera columns, and meta/info.json features.
"""
import json
import shutil
import sys
from pathlib import Path

import av
import pandas as pd

KEY_A = "observation.images.fpv"
KEY_B = "observation.images.left"


def first_frame(mp4, out_jpg):
    with av.open(str(mp4)) as c:
        for frame in c.decode(c.streams.video[0]):
            frame.to_image().save(out_jpg)
            return True
    return False


def inspect(root: Path):
    name = root.name
    for key, tag in [(KEY_A, "fpv"), (KEY_B, "left")]:
        mp4s = sorted((root / "videos" / key).rglob("*.mp4"))
        if not mp4s:
            print(f"{name}/{tag}: NO VIDEOS FOUND")
            continue
        out = Path.home() / f"check_{name}_{tag}.jpg"
        first_frame(mp4s[0], out)
        print(f"{name}/{tag}: {len(mp4s)} file(s) -> {out}")


def swap(root: Path):
    name = root.name
    vids = root / "videos"
    a, b = vids / KEY_A, vids / KEY_B
    assert a.is_dir() and b.is_dir(), f"{name}: missing video dirs"

    # 1. swap video directories
    tmp = vids / "__swap_tmp__"
    a.rename(tmp)
    b.rename(a)
    tmp.rename(b)
    print(f"{name}: video dirs swapped")

    # 2. swap stats.json entries
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    if KEY_A in stats and KEY_B in stats:
        stats[KEY_A], stats[KEY_B] = stats[KEY_B], stats[KEY_A]
        stats_path.write_text(json.dumps(stats, indent=4))
        print(f"{name}: stats.json swapped")

    # 3. swap per-camera columns in episodes metadata parquet(s)
    n_files = 0
    for pq in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(pq)
        cols_a = [c for c in df.columns if KEY_A in c]
        swapped = []
        for ca in cols_a:
            cb = ca.replace(KEY_A, KEY_B)
            if cb in df.columns:
                df[ca], df[cb] = df[cb].copy(), df[ca].copy()
                swapped.append(ca)
        if swapped:
            df.to_parquet(pq)
            n_files += 1
    print(f"{name}: episode metadata swapped in {n_files} parquet file(s)")

    # 4. swap feature entries in info.json
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    feats = info.get("features", {})
    if KEY_A in feats and KEY_B in feats:
        feats[KEY_A], feats[KEY_B] = feats[KEY_B], feats[KEY_A]
        info_path.write_text(json.dumps(info, indent=4))
        print(f"{name}: info.json features swapped")

    print(f"{name}: DONE\n")


if __name__ == "__main__":
    mode, *roots = sys.argv[1:]
    assert mode in ("inspect", "swap") and roots, __doc__
    for r in roots:
        root = Path(r)
        assert (root / "meta" / "info.json").exists(), f"not a dataset root: {root}"
        if mode == "inspect":
            inspect(root)
        else:
            # safety: refuse to swap twice by accident -- marker file
            marker = root / "meta" / ".cameras_swapped"
            if marker.exists():
                print(f"{root.name}: ALREADY SWAPPED (marker exists), skipping")
                continue
            swap(root)
            marker.touch()
