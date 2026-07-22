#!/usr/bin/env python3
"""G0 validate: RoboCasa paper zarr = 200 eps (50H+150M), Da=12, required keys, no NaN."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "robocasa"

TASKS = (
    "close_drawer",
    "coffee_press_button",
    "turn_off_microwave",
    "turn_off_sink_faucet",
)

REQUIRED_DATA_KEYS = (
    "action",
    "robot0_agentview_left_rgb",
    "robot0_agentview_right_rgb",
    "robot0_eye_in_hand_rgb",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def validate_one(task: str, n_demo: int = 200) -> bool:
    zpath = DATA / f"{task}_N{n_demo}.zarr"
    print(f"=== {task} ===")
    if not zpath.is_dir():
        print(f"  FAIL missing zarr: {zpath}")
        return False
    z = zarr.open(str(zpath), mode="r")
    ends = z["meta"]["episode_ends"][:]
    n_ep = len(ends)
    n_steps = int(ends[-1])
    action = z["data"]["action"]
    ok = True
    if n_ep != n_demo:
        print(f"  FAIL episodes={n_ep} expected {n_demo}")
        ok = False
    if int(action.shape[-1]) != 12:
        print(f"  FAIL action_dim={action.shape[-1]} expected 12")
        ok = False
    missing = [k for k in REQUIRED_DATA_KEYS if k not in z["data"]]
    if missing:
        print(f"  FAIL missing keys: {missing}")
        ok = False
    # finite action sample
    idx = np.linspace(0, n_steps - 1, num=min(256, n_steps), dtype=np.int64)
    sample = action.get_orthogonal_selection((idx, slice(None)))
    if not np.all(np.isfinite(sample)):
        print("  FAIL NaN/Inf in action sample")
        ok = False
    src = zpath / "ROBOCASA_SOURCE.txt"
    if not src.is_file():
        print("  WARN missing ROBOCASA_SOURCE.txt provenance")
    else:
        print(f"  provenance: {src.read_text(encoding='utf-8').splitlines()[:4]}")
    print(f"  zarr episodes={n_ep} steps={n_steps} action={action.shape}")
    if ok:
        print("  OK")
    return ok


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task",
        action="append",
        choices=list(TASKS),
        help="Validate only these tasks (repeatable). Default: all.",
    )
    args = p.parse_args()
    tasks = tuple(args.task) if args.task else TASKS
    ok = True
    for task in tasks:
        ok = validate_one(task) and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
