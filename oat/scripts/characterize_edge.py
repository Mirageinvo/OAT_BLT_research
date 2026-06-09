"""
WHERE are the fate-deciding edge states? (NO sim, instant)

From a `--bon_isolate` npz, split states by recoverability into doomed (p<lo),
edge (lo<=p<=hi), safe (p>hi), and compare their logged features (step in episode,
gripper openness, eef velocity, MuJoCo contacts, grip-change rate). Answers: are
edge states at the grasp moment, or spread / somewhere non-obvious?

Run:
  cd oat && uv run python scripts/characterize_edge.py -i my_datasets/iso_big.npz
"""
if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).parent.parent))

import click
import numpy as np


@click.command()
@click.option('-i', '--input', 'inp', required=True)
@click.option('--lo', default=0.35, type=float, help='edge band lower (recoverability)')
@click.option('--hi', default=0.65, type=float, help='edge band upper')
@click.option('--rkey', default='baseline', type=click.Choice(['baseline', 'baseline_eval']),
              help='recoverability proxy (baseline = all-M mean, cleaner)')
def main(inp, lo, hi, rkey):
    d = np.load(inp, allow_pickle=True)
    if 'heldout' not in d.files:
        print(f"not an isolate file. keys={d.files}"); return
    r = d[rkey].astype(float)
    crit = d['heldout'].astype(float) - d['baseline_eval'].astype(float)
    feats = {k: d[k].astype(float) for k in
             ['step', 'gripper_open', 'eef_vel', 'ncon', 'grip_will_change'] if k in d.files}

    doomed = r < lo
    edge = (r >= lo) & (r <= hi)
    safe = r > hi
    groups = [('doomed(p<%.2f)' % lo, doomed), ('EDGE[%.2f,%.2f]' % (lo, hi), edge),
              ('safe(p>%.2f)' % hi, safe)]

    print(f"\n=== where are edge states?  ({inp}, n={len(r)}, rkey={rkey}) ===")
    print(f"counts: doomed={int(doomed.sum())}  edge={int(edge.sum())}  safe={int(safe.sum())}")
    print(f"mean criticality: doomed={crit[doomed].mean():+.3f}  "
          f"EDGE={crit[edge].mean():+.3f}  safe={crit[safe].mean():+.3f}\n")

    cols = list(feats.keys())
    print(f"  {'group':<16} | " + " | ".join(f"{c:>13}" for c in cols))
    for name, m in groups:
        if m.sum() == 0:
            print(f"  {name:<16} | n=0"); continue
        cells = []
        for c in cols:
            v = feats[c][m]
            cells.append(f"{v.mean():>7.3f}±{v.std()/max(np.sqrt(len(v)),1):<5.3f}")
        print(f"  {name:<16} | " + " | ".join(cells))

    print("\nREAD: if EDGE row differs from doomed/safe on some feature (e.g. step, "
          "gripper_open) -> edge states have a physical signature (detectable). "
          "If EDGE ~= others on all features -> recoverability is NOT readable from "
          "these features (need rich obs / value head; obs-wall risk).")
    print("\nDONE")


if __name__ == '__main__':
    main()
