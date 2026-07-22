"""
Convert a RoboTwin task's demos -> OAT Zarr. Mirror of scripts/convert_libero_dataset.py.

Run:
  cd oat && uv run python scripts/convert_robotwin_dataset.py \
      --src data/<task> --task dual_bottles_pick_easy -n 500 --format hdf5
Output: data/robotwin/<task>_N<n_demo>.zarr  (matches the robotwin task configs' zarr_path).
See oat/env/robotwin/dataset_conversion.py for the RoboTwin-specific TODOs (action layout,
camera keys, proprio, orientation, LeRobot mapping) — VERIFY before a full run.
"""
if __name__ == "__main__":
    import sys, os, pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import pathlib

import click
import zarr

from oat.env.robotwin.dataset_conversion import (
    convert_robotwin_hdf5_to_zarr, convert_robotwin_lerobot_to_zarr)


@click.command()
@click.option('--src', required=True, help='HDF5 dir (format=hdf5) or LeRobot dir (format=lerobot)')
@click.option('--task', required=True, help='task name -> output zarr filename')
@click.option('--root_dir', default='data/robotwin')
@click.option('--format', 'fmt', type=click.Choice(['hdf5', 'lerobot']), default='hdf5')
@click.option('--task_uid', default=0, type=int)
@click.option('-n', '--num_sample_demo', type=int, default=None)
@click.option('--flip_images', is_flag=True, default=False)
@click.option('--image_size', default=128, type=int)
def main(src, task, root_dir, fmt, task_uid, num_sample_demo, flip_images, image_size):
    conv = convert_robotwin_hdf5_to_zarr if fmt == 'hdf5' else convert_robotwin_lerobot_to_zarr
    rb = conv(src, task_uid=task_uid, sample_ndemo=num_sample_demo, flip_images=flip_images,
              image_size=image_size)
    n_demo = rb.n_episodes
    save_path = f"{root_dir}/{task}_N{n_demo}.zarr"
    if os.path.exists(save_path):
        os.system(f"rm -rf {save_path}")
    pathlib.Path(save_path).mkdir(parents=True)
    rb.save_to_path(save_path, compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=1))
    print(f"Saved -> {save_path}")


if __name__ == "__main__":
    main()
