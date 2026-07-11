#!/usr/bin/env python3
"""Select best checkpoint in a run dir by metric encoded in filename.

Examples:
  python scripts/select_best_ckpt_by_name.py --run-dir output/... --metric mse --mode min
  python scripts/select_best_ckpt_by_name.py --run-dir output/... --metric sr  --mode max
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metric", type=str, required=True, help="e.g. mse or sr")
    parser.add_argument("--mode", choices=["min", "max"], required=True)
    parser.add_argument("--top", type=int, default=1, help="Print top-K paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = args.run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    pattern = re.compile(rf"{re.escape(args.metric)}-([0-9.]+)")
    parsed: list[tuple[float, Path]] = []
    for p in ckpt_dir.glob("ep-*_" + args.metric + "-*.ckpt"):
        m = pattern.search(p.name)
        if not m:
            continue
        val = float(m.group(1).rstrip("."))
        parsed.append((val, p))

    if not parsed:
        raise SystemExit(f"No checkpoints with metric '{args.metric}' in {ckpt_dir}")

    parsed.sort(key=lambda x: x[0], reverse=(args.mode == "max"))
    for _, p in parsed[: args.top]:
        print(p)


if __name__ == "__main__":
    main()
