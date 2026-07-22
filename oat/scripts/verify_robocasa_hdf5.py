#!/usr/bin/env python3
"""Quick HDF5 smoke test after Box download (has data/, demo count)."""

from __future__ import annotations

import pathlib
import sys

import h5py


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_robocasa_hdf5.py PATH.hdf5", file=sys.stderr)
        return 2
    path = pathlib.Path(sys.argv[1])
    if not path.is_file():
        print(f"[err] missing {path}", file=sys.stderr)
        return 1
    try:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                print("[err] missing top-level group 'data'", file=sys.stderr)
                return 1
            demos = list(f["data"].keys())
    except OSError as e:
        print(f"[err] h5py open failed: {e}", file=sys.stderr)
        return 1
    if not demos:
        print("[err] zero demos in data/", file=sys.stderr)
        return 1
    print(f"OK {len(demos)} demos in {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
