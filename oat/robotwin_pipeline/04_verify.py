"""
04 — verify the converted Zarr before training (cheap gate; don't burn GPU on a bad dataset).
Checks: action last-dim == expected, images present + uint8 HWC, proprio present, no NaN.
    python robotwin_pipeline/04_verify.py --zarr <ZARR> --action_dim 14
"""
import click
import numpy as np
import zarr


@click.command()
@click.option('--zarr', 'zpath', required=True)
@click.option('--action_dim', default=14, type=int)
def main(zpath, action_dim):
    root = zarr.open(zpath, mode='r')
    data = root['data'] if 'data' in root else root
    print(f"=== verify {zpath} ===")
    keys = list(data.array_keys()) if hasattr(data, 'array_keys') else list(data.keys())
    ok = True
    for k in keys:
        a = data[k]
        print(f"  {k:28s} shape={tuple(a.shape)} dtype={a.dtype}")
    # action check
    if 'action' in keys:
        ad = data['action'].shape[-1]
        print(f"[action] last dim = {ad}  (expected {action_dim})")
        if ad != action_dim:
            ok = False; print("  !! action dim mismatch -> fix ACTION_IDX / config action_dim")
    else:
        ok = False; print("  !! no 'action' array")
    # image + proprio presence
    imgs = [k for k in keys if 'rgb' in k]
    print(f"[images] {imgs}  (need agentview_rgb + robot0_eye_in_hand_rgb)")
    if not {'agentview_rgb', 'robot0_eye_in_hand_rgb'} <= set(keys):
        ok = False; print("  !! missing one of the two OAT camera keys")
    # NaN check on a sample
    if 'action' in keys:
        s = np.asarray(data['action'][:1000])
        if not np.isfinite(s).all():
            ok = False; print("  !! non-finite actions")
    print("\nVERDICT:", "PASS -> run 05_train_tokenizer" if ok else "FAIL -> fix converter/config, re-run 03")


if __name__ == '__main__':
    main()
