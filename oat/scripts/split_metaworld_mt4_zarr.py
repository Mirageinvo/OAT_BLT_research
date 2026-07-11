#!/usr/bin/env python3
"""
Split a multitask MetaWorld MT4 zarr into four single-task zarr datasets.

Input assumptions (true for scripts/gen_metaworld_data.py):
- MT4 episodes are accepted in round-robin task order:
  [box-close, coffee-pull, disassemble, stick-pull], repeated.
- total episodes = num_tasks * episodes_per_task

Output:
- data/metaworld/box-close_N50.zarr
- data/metaworld/coffee-pull_N50.zarr
- data/metaworld/disassemble_N50.zarr
- data/metaworld/stick-pull_N50.zarr
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr

from oat.common.replay_buffer import ReplayBuffer
from oat.env.metaworld.factory import get_subtasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/metaworld/mt4_N50.zarr"),
        help="Source MT4 zarr path.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="mt4",
        help="MetaWorld multitask suite name (default: mt4).",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=50,
        help="Expected successful episodes per subtask.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/metaworld"),
        help="Directory where per-task zarrs will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-task datasets if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subtasks = get_subtasks(args.task_name)
    if len(subtasks) <= 1:
        raise ValueError(f"{args.task_name} is not multitask: subtasks={subtasks}")

    if not args.src.is_dir():
        raise FileNotFoundError(f"Source zarr not found: {args.src}")

    src_rb = ReplayBuffer.copy_from_path(str(args.src), store=None)
    n_episodes = src_rb.n_episodes
    expected = len(subtasks) * args.episodes_per_task
    if n_episodes != expected:
        raise ValueError(
            f"Unexpected episode count in {args.src}: {n_episodes} (expected {expected})"
        )

    counts = np.zeros(len(subtasks), dtype=np.int64)
    buffers = [ReplayBuffer.create_empty_numpy() for _ in subtasks]

    for ep_idx in range(n_episodes):
        task_idx = ep_idx % len(subtasks)
        episode = src_rb.get_episode(ep_idx, copy=True)
        buffers[task_idx].add_episode(episode)
        counts[task_idx] += 1

    if not np.all(counts == args.episodes_per_task):
        raise RuntimeError(
            f"Split mismatch: got counts={counts.tolist()} expected={args.episodes_per_task}"
        )

    compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=1)
    args.out_root.mkdir(parents=True, exist_ok=True)

    for task_name, buff in zip(subtasks, buffers):
        out_path = args.out_root / f"{task_name}_N{args.episodes_per_task}.zarr"
        if out_path.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"{out_path} exists; pass --overwrite to replace it."
                )
            import shutil

            shutil.rmtree(out_path)

        buff.update_meta(
            {
                "source_suite": np.array(args.task_name),
                "source_zarr": np.array(str(args.src)),
                "task_name": np.array(task_name),
                "num_episodes": np.array(buff.n_episodes, dtype=np.int64),
            }
        )
        buff.save_to_path(str(out_path), compressors=compressor)
        print(f"[OK] {task_name}: {buff.n_episodes} episodes -> {out_path}")


if __name__ == "__main__":
    main()
