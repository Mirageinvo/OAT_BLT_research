#!/usr/bin/env python3
"""Print path to the best MT4 tokenizer checkpoint (lowest test_reconst_mse)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _parse_mse_from_name(name: str) -> float | None:
    m = re.search(r"mse-([0-9.]+)", name)
    if not m:
        return None
    return float(m.group(1).rstrip("."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="output/20260707/124135_train_oattok_mw-mt4_N50",
        help="Hydra run directory for train_oattok mw-mt4_N50.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_path = run_dir / "logs.json"
    ckpt_dir = run_dir / "checkpoints"

    best_ep: int | None = None
    best_mse: float | None = None
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            mse = row.get("test_reconst_mse")
            ep = row.get("epoch")
            if mse is None or ep is None:
                continue
            mse_f = float(mse)
            if best_mse is None or mse_f < best_mse:
                best_mse = mse_f
                best_ep = int(ep)

    if best_ep is None:
        raise SystemExit(f"No test_reconst_mse found in {log_path}")

    # Prefer the saved top-k file for that epoch; fall back to nearest name match.
    candidates = sorted(ckpt_dir.glob(f"ep-{best_ep:04d}_mse-*.ckpt"))
    if not candidates:
        parsed = []
        for p in ckpt_dir.glob("ep-*_mse-*.ckpt"):
            mse = _parse_mse_from_name(p.name)
            if mse is not None:
                parsed.append((mse, p))
        if not parsed:
            raise SystemExit(f"No tokenizer checkpoints in {ckpt_dir}")
        best_path = min(parsed, key=lambda x: x[0])[1]
    else:
        best_path = candidates[0]

    print(best_path)
    if best_mse is not None:
        import sys
        print(f"best epoch={best_ep} test_reconst_mse={best_mse:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
