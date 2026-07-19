"""Convert RoboCasa v0.2 official HDF5 (50H+150M) → OAT zarr. See ROBOCASA.md G0."""

from __future__ import annotations

import pathlib
import shutil
import sys

import click
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oat.env.robocasa.dataset_conversion import (  # noqa: E402
    PAPER_N_HUMAN,
    PAPER_N_MACHINE,
    PAPER_N_TOTAL,
    TASK_ID_TO_PASCAL,
    convert_robocasa_pair_to_zarr,
)


@click.command()
@click.option(
    "--task",
    type=click.Choice(sorted(TASK_ID_TO_PASCAL.keys())),
    required=True,
)
@click.option(
    "--hdf5-root",
    type=click.Path(path_type=pathlib.Path),
    default=ROOT / "data" / "robocasa" / "hdf5",
)
@click.option(
    "--out-root",
    type=click.Path(path_type=pathlib.Path),
    default=ROOT / "data" / "robocasa",
)
@click.option(
    "--human-hdf5",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Override human HDF5 path.",
)
@click.option(
    "--mg-hdf5",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Override MG HDF5 (prefer slim M150 file after download).",
)
@click.option("--n-human", type=int, default=PAPER_N_HUMAN, show_default=True)
@click.option("--n-machine", type=int, default=PAPER_N_MACHINE, show_default=True)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Subsample seed for H/M selection (paper collect seed 0).",
)
@click.option("--force/--no-force", default=False, help="Overwrite existing zarr.")
def main(
    task: str,
    hdf5_root: pathlib.Path,
    out_root: pathlib.Path,
    human_hdf5: pathlib.Path | None,
    mg_hdf5: pathlib.Path | None,
    n_human: int,
    n_machine: int,
    seed: int,
    force: bool,
) -> None:
    pascal = TASK_ID_TO_PASCAL[task]
    human = human_hdf5 or (
        hdf5_root
        / pascal
        / "human"
        / "demo_gentex_im128_randcams.hdf5"
    )
    # Prefer paper slim M150 if present (full mg_im ≈ 24GB).
    slim = hdf5_root / pascal / "mg" / f"demo_gentex_im128_randcams_M150_s{seed}.hdf5"
    mg = mg_hdf5 or (
        slim
        if slim.is_file()
        else hdf5_root / pascal / "mg" / "demo_gentex_im128_randcams.hdf5"
    )
    if not human.is_file():
        raise FileNotFoundError(f"Missing human HDF5: {human}")
    if not mg.is_file():
        raise FileNotFoundError(f"Missing mg HDF5: {mg}")

    n_total = n_human + n_machine
    out = out_root / f"{task}_N{n_total}.zarr"
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} exists (pass --force to overwrite)")
        shutil.rmtree(out)

    out_root.mkdir(parents=True, exist_ok=True)
    compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=1)
    stats = convert_robocasa_pair_to_zarr(
        human_hdf5=str(human),
        mg_hdf5=str(mg),
        zarr_path=str(out),
        n_human=n_human,
        n_machine=n_machine,
        seed=seed,
        compressor=compressor,
    )
    assert stats["episodes"] == n_total
    assert stats["action_dim"] == 12
    print(f"OK paper mix: {task} → {out}")


if __name__ == "__main__":
    main()
