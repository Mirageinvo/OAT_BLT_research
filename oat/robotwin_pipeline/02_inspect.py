"""
02 — inspect ONE downloaded RoboTwin episode and print exactly what to paste into the converter
TODOs and the task configs (action_dim, camera keys, proprio keys). Run:
    python robotwin_pipeline/02_inspect.py --src <SRC_DIR> --format hdf5
This removes the guesswork: copy the printed ACTION_IDX / CAM_* / proprio dims into
oat/env/robotwin/dataset_conversion.py and the config action_dim / shape_meta.
"""
import sys, glob, os
import click


def dump_hdf5(path):
    import h5py
    import numpy as np
    print(f"\n=== HDF5: {path} ===")
    rows = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            rows.append((name, tuple(obj.shape), str(obj.dtype)))
    with h5py.File(path, 'r') as f:
        f.visititems(visit)
    imgs, actions, states = [], [], []
    for name, shape, dt in rows:
        print(f"  {name:45s} shape={shape} dtype={dt}")
        if len(shape) >= 3 and shape[-1] in (3, 4) and 'uint' in dt:
            imgs.append((name, shape))
        elif 'action' in name.lower():
            actions.append((name, shape))
        elif len(shape) == 2 and shape[-1] <= 32:
            states.append((name, shape))
    print("\n--- SUGGESTED MAPPING (paste into dataset_conversion.py / configs) ---")
    if actions:
        an, ash = actions[0]
        print(f"  ACTION_KEY = '{an}'   # per-step action dim = {ash[-1]}  -> set config action_dim")
        print(f"  ACTION_IDX = list(range(0, {ash[-1]}))   # verify this is the 14 bimanual dims")
    print(f"  image-like keys (pick 2 cameras): {[n for n,_ in imgs]}")
    print(f"  state-like keys (proprio, dual-arm): {[(n,s[-1]) for n,s in states]}")


def dump_lerobot(src):
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:
        print(f"lerobot not importable: {e}"); return
    ds = LeRobotDataset(src)
    print("\n=== LeRobot features ===")
    for k, v in ds.features.items():
        print(f"  {k:45s} {v}")
    print("\n--- map action / two image features / proprio into the converter (D1/D3/D4) ---")


@click.command()
@click.option('--src', required=True)
@click.option('--format', 'fmt', type=click.Choice(['hdf5', 'lerobot']), default='hdf5')
def main(src, fmt):
    if fmt == 'hdf5':
        files = sorted(glob.glob(os.path.join(src, '**', '*.hdf5'), recursive=True))
        if not files:
            print(f"no .hdf5 under {src}"); sys.exit(1)
        dump_hdf5(files[0])
    else:
        dump_lerobot(src)
    print("\nNEXT: paste the mapping into oat/env/robotwin/dataset_conversion.py (ACTION_IDX, "
          "CAM_AGENTVIEW, CAM_EYE, proprio keys) and the two robotwin configs (action_dim, "
          "shape_meta, camera_names). THEN fill oat/env/robotwin/env.py + copy the runner. THEN run 03.")


if __name__ == '__main__':
    main()
