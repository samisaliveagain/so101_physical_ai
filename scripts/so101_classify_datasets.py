#!/usr/bin/env python3
"""Print task prompt, episode count, and swap-fix marker for each dataset."""
import json
import sys
from pathlib import Path

import pandas as pd

base = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
for root in sorted(base.glob("world-models*"), key=lambda p: (len(p.name), p.name)):
    info = json.loads((root / "meta" / "info.json").read_text())
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    # task strings may live in the index or in any object-typed column
    strings = [str(v) for v in tasks.index if isinstance(v, str)]
    for col in tasks.columns:
        strings += [str(v) for v in tasks[col] if isinstance(v, str)]
    if not strings:  # fall back to raw dump
        strings = [tasks.to_dict("records")]
    mark = "Y" if (root / "meta" / ".cameras_swapped").exists() else "-"
    kind = "?"
    joined = " ".join(map(str, strings)).lower()
    if "hexagon" in joined or "grab" in joined:
        kind = "VLA"
    elif "explor" in joined or "world" in joined:
        kind = "WM "
    print(f"{root.name:<16} eps={info['total_episodes']:<4} fixed={mark} [{kind}] {strings}")
