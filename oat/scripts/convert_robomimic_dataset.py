"""
Convert RoboMimic HDF5 datasets to OAT zarr format.
"""

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import pathlib
import shutil

import click
import h5py
import zarr

from oat.common.input_util import wait_user_input
from oat.env.robomimic.dataset_conversion import (
    DEFAULT_REQUIRED_OBS_KEYS,
    convert_robomimic_hdf5_to_zarr_streaming,
    infer_task_name,
)


@click.command()
@click.option("--root_dir", type=str, default="data/robomimic")
@click.option("--hdf5_dir_name", type=str, default="hdf5_datasets")
@click.option("-n", "--num_sample_demo", type=int, default=None)
@click.option("--seed", type=int, default=42)
@click.option("--compression_level", type=int, default=5, show_default=True)
@click.option("--chunk_size", type=int, default=1024, show_default=True)
@click.option("--verify_sample_size", type=int, default=128, show_default=True)
@click.option(
    "--required_obs_key",
    type=str,
    multiple=True,
    default=DEFAULT_REQUIRED_OBS_KEYS,
    help="Required obs keys in each demo['obs']. Repeat this option to override defaults.",
)
@click.option(
    "--hdf5",
    "hdf5_filter",
    type=str,
    default=None,
    help="Convert only this HDF5 (path or basename under hdf5_dir).",
)
@click.option(
    "--skip-existing/--overwrite",
    default=False,
    help="Skip existing zarr (non-interactive). Default: prompt on conflict.",
)
def convert_all_robomimic_datasets(
    root_dir: str,
    hdf5_dir_name: str,
    num_sample_demo: int,
    seed: int,
    compression_level: int,
    chunk_size: int,
    verify_sample_size: int,
    required_obs_key: tuple[str, ...],
    hdf5_filter: str | None,
    skip_existing: bool,
):
    hdf5_root = pathlib.Path(root_dir) / hdf5_dir_name
    if hdf5_filter is not None:
        p = pathlib.Path(hdf5_filter)
        if not p.is_absolute():
            p = hdf5_root / hdf5_filter
        if not p.exists():
            raise FileNotFoundError(p)
        hdf5_paths = [p]
    else:
        hdf5_paths = sorted(hdf5_root.glob("*.hdf5"))
    if not hdf5_paths:
        raise FileNotFoundError(f"No .hdf5 files found in {hdf5_root}")

    # Prefer multi-human (mh) dumps when both mh and ph exist for the same task.
    task_to_path: dict[str, pathlib.Path] = {}
    for hdf5_path in hdf5_paths:
        task_name = infer_task_name(str(hdf5_path))
        stem = hdf5_path.stem.lower()
        prev = task_to_path.get(task_name)
        if prev is None:
            task_to_path[task_name] = hdf5_path
            continue
        prev_mh = "mh" in prev.stem.lower()
        cur_mh = "mh" in stem
        if cur_mh and not prev_mh:
            task_to_path[task_name] = hdf5_path
    hdf5_paths = sorted(task_to_path.values())

    for hdf5_path in hdf5_paths:
        hdf5_path_str = str(hdf5_path)
        print(f"Converting {hdf5_path_str}...")

        task_name = infer_task_name(hdf5_path_str)
        with h5py.File(hdf5_path_str, "r") as f:
            all_demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
            expected_n_demo = len(all_demo_keys) if num_sample_demo is None else min(num_sample_demo, len(all_demo_keys))
        save_path = pathlib.Path(root_dir) / f"{task_name}_N{expected_n_demo}.zarr"

        if save_path.exists():
            if skip_existing:
                print(f"Skip existing export: {save_path}")
                continue
            keypress = wait_user_input(
                valid_input=lambda key: key in ["", "y", "n"],
                prompt=f"{save_path} already exists. Overwrite? [y/`n`]: ",
                default="n",
            )
            if keypress == "n":
                print("Skip existing export.")
                continue
            shutil.rmtree(save_path)

        compressor = zarr.Blosc(cname="zstd", clevel=compression_level, shuffle=1)
        stats = convert_robomimic_hdf5_to_zarr_streaming(
            hdf5_path=hdf5_path_str,
            zarr_path=str(save_path),
            sample_ndemo=num_sample_demo,
            required_obs_keys=required_obs_key,
            seed=seed,
            compressor=compressor,
            chunk_size=chunk_size,
            verify_sample_size=verify_sample_size,
        )
        print(
            f"Verification passed: episodes={stats['episodes']}, "
            f"steps={stats['steps']}, action_dim={stats['action_dim']}"
        )

    print("All done.")


if __name__ == "__main__":
    convert_all_robomimic_datasets()
