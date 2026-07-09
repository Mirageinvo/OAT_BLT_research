#!/usr/bin/env python3
"""Quick check: mh image HDF5 + N200 zarr per RoboMimic task (paper protocol)."""
from pathlib import Path

import h5py
import zarr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "robomimic"

TASKS = (
    ("lift", "lift_mh_image.hdf5"),
    ("can", "can_mh_image.hdf5"),
    ("square", "square_mh_image.hdf5"),
)


def main() -> None:
    ok = True
    for task, h5_name in TASKS:
        h5 = DATA / "hdf5_datasets" / h5_name
        zpath = DATA / f"{task}_N200.zarr"
        print(f"=== {task} ===")
        if not h5.is_file():
            print(f"  FAIL missing HDF5: {h5}")
            ok = False
            continue
        with h5py.File(h5, "r") as f:
            demos = [k for k in f["data"].keys() if k.startswith("demo_")]
            img = f["data"][demos[0]]["obs"]["agentview_image"]
            print(f"  HDF5: {h5_name} demos={len(demos)} img={img.shape} dtype={img.dtype}")
        if not zpath.is_dir():
            print(f"  FAIL missing zarr: {zpath}")
            ok = False
            continue
        z = zarr.open(str(zpath), mode="r")
        n_ep = len(z["meta"]["episode_ends"][:])
        n_steps = int(z["meta"]["episode_ends"][-1])
        print(f"  ZARR: {zpath.name} episodes={n_ep} steps={n_steps}")
        if n_ep != 200:
            print(f"  WARN expected 200 episodes (paper num_demo=200), got {n_ep}")
            ok = False
        else:
            print("  OK")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
