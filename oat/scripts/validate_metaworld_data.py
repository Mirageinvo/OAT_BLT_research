#!/usr/bin/env python3
"""Validate MetaWorld MT4 Zarr generated for the OAT paper baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


EXPECTED_KEYS = {
    "action": (None, 4),
    "corner_rgb": (None, 128, 128, 3),
    "corner2_rgb": (None, 128, 128, 3),
    "corner3_rgb": (None, 128, 128, 3),
    "behindGripper_rgb": (None, 128, 128, 3),
    "agent_pos": (None, 9),
}


def _shape_matches(actual: tuple[int, ...], expected: tuple[int | None, ...]) -> bool:
    return len(actual) == len(expected) and all(
        e is None or a == e for a, e in zip(actual, expected)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "zarr_path",
        nargs="?",
        default="data/metaworld/mt4_N50.zarr",
        help="Path to generated MetaWorld MT4 zarr.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument(
        "--require-subtask-counts",
        action="store_true",
        help=(
            "Fail if meta/subtask_counts is missing. "
            "Recommended for multitask datasets (e.g. mt4_N50.zarr)."
        ),
    )
    args = parser.parse_args()

    path = Path(args.zarr_path)
    if not path.is_dir():
        raise FileNotFoundError(path)

    root = zarr.open(str(path), mode="r")
    data = root["data"]
    meta = root["meta"]
    episode_ends = np.asarray(meta["episode_ends"][:])
    expected_episodes = args.episodes_per_task * args.num_tasks

    print(f"zarr: {path}")
    print(f"episodes: {len(episode_ends)} expected={expected_episodes}")
    if len(episode_ends) != expected_episodes:
        raise AssertionError(f"Expected {expected_episodes} episodes, got {len(episode_ends)}")

    for key, expected_shape in EXPECTED_KEYS.items():
        if key not in data:
            raise KeyError(f"Missing data/{key}")
        arr = data[key]
        print(f"data/{key}: shape={arr.shape} dtype={arr.dtype}")
        if not _shape_matches(arr.shape, expected_shape):
            raise AssertionError(f"Unexpected shape for {key}: {arr.shape}")
        if arr.shape[0] != episode_ends[-1]:
            raise AssertionError(f"{key} length {arr.shape[0]} != episode_ends[-1] {episode_ends[-1]}")

    expect_multitask = args.num_tasks > 1
    require_subtask_counts = args.require_subtask_counts or expect_multitask

    if "subtask_counts" in meta:
        counts = np.asarray(meta["subtask_counts"][:], dtype=np.int64)
        print(f"subtask_counts: {counts.tolist()}")
        if expect_multitask:
            expected_counts = np.full(args.num_tasks, args.episodes_per_task, dtype=np.int64)
            if not np.array_equal(counts, expected_counts):
                raise AssertionError(
                    f"Expected subtask_counts {expected_counts.tolist()}, got {counts.tolist()}"
                )
        else:
            print("subtask_counts: present (single-task file may omit this key; this is fine)")
    else:
        if require_subtask_counts:
            raise AssertionError(
                "subtask_counts: missing for multitask validation. "
                "Pass --num-tasks 1 for single-task files or regenerate multitask data."
            )
        print("subtask_counts: missing (expected for single-task datasets)")

    if "num_episodes" in meta:
        num_episodes_meta = int(np.asarray(meta["num_episodes"][()]).item())
        print(f"meta/num_episodes: {num_episodes_meta}")
        if num_episodes_meta != expected_episodes:
            raise AssertionError(
                f"meta/num_episodes={num_episodes_meta} does not match expected {expected_episodes}"
            )

    print("OK")


if __name__ == "__main__":
    main()
