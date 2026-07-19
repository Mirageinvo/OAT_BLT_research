"""Subsample MimicGen RoboCasa HDF5 → slim file (paper: 150 demos, seed 0).

Full mg_im is ~24GB; we only need 150 demos for the OAT paper mix.
"""

from __future__ import annotations

import pathlib
import sys

import click
import h5py
import numpy as np
import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oat.env.robocasa.dataset_conversion import (  # noqa: E402
    PAPER_N_MACHINE,
    _select_demo_keys,
)


def copy_selected_demos(
    src_path: pathlib.Path,
    dst_path: pathlib.Path,
    n: int,
    seed: int,
) -> list[str]:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        if "data" not in src:
            raise KeyError(f"{src_path} missing 'data'")
        keys = _select_demo_keys(src["data"], n=n, seed=seed)
        dst_data = dst.create_group("data")
        # copy top-level attrs
        for ak, av in src["data"].attrs.items():
            dst_data.attrs[ak] = av
        for k in tqdm.tqdm(keys, desc=f"slim→{dst_path.name}"):
            src.copy(src["data"][k], dst_data, name=k)
        # optional: copy mask / other groups if present (not required for convert)
        for gname in src.keys():
            if gname == "data":
                continue
            # skip large unused groups
            if gname in ("mask",):
                src.copy(src[gname], dst, name=gname)

    return keys


@click.command()
@click.option("--src", type=click.Path(exists=True, path_type=pathlib.Path), required=True)
@click.option("--dst", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--n", type=int, default=PAPER_N_MACHINE, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--force/--no-force", default=False)
def main(src: pathlib.Path, dst: pathlib.Path, n: int, seed: int, force: bool) -> None:
    if dst.exists() and not force:
        raise FileExistsError(f"{dst} exists (pass --force)")
    if dst.exists():
        dst.unlink()
    keys = copy_selected_demos(src, dst, n=n, seed=seed)
    sz = dst.stat().st_size / 1e9
    print(f"OK {len(keys)} demos → {dst} ({sz:.2f} GB)")
    print(f"first={keys[0]} last={keys[-1]} seed={seed}")


if __name__ == "__main__":
    main()
