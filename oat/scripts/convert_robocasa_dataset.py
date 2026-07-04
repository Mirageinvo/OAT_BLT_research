"""
Convert a RoboCasa task HDF5 (robomimic-format, WITH camera obs) -> OAT Zarr.
Mirror of scripts/convert_libero_dataset.py.

Run:
  cd oat && uv run python scripts/convert_robocasa_dataset.py \
      --hdf5_path <path/to/robocasa_task_demo.hdf5> --task CloseDrawer -n 500

Output: data/robocasa/<task>_N<n_demo>.zarr  (matches the zarr_path in the robocasa task configs).
See oat/env/robocasa/dataset_conversion.py for the RoboCasa-specific TODOs (action slicing,
camera keys, quat convention, image flip) — VERIFY them before a full run.
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

from oat.env.robocasa.dataset_conversion import convert_robocasa_hdf5_to_zarr


@click.command()
@click.option('--hdf5_path', required=True, help='RoboCasa task HDF5 (robomimic format, with images)')
@click.option('--task', required=True, help='task name -> used in the output zarr filename')
@click.option('--root_dir', default='data/robocasa')
@click.option('--task_uid', default=0, type=int, help='constant task_uid for a single task')
@click.option('-n', '--num_sample_demo', type=int, default=None, help='subsample N demos (default all)')
@click.option('--flip_images', is_flag=True, default=False, help='flip camera images vertically (TODO #4)')
def main(hdf5_path, task, root_dir, task_uid, num_sample_demo, flip_images):
    rb = convert_robocasa_hdf5_to_zarr(
        hdf5_path=hdf5_path, task_uid=task_uid,
        sample_ndemo=num_sample_demo, flip_images=flip_images,
    )
    n_demo = rb.n_episodes
    save_path = f"{root_dir}/{task}_N{n_demo}.zarr"
    if os.path.exists(save_path):
        os.system(f"rm -rf {save_path}")
    pathlib.Path(save_path).mkdir(parents=True)
    compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=1)
    rb.save_to_path(save_path, compressor=compressor)
    print(f"Saved -> {save_path}")


if __name__ == "__main__":
    main()
