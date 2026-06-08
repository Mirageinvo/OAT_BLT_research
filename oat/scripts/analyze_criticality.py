"""
Inverted-U test (NO sim, instant): from a `--bon_isolate` npz, does per-state
criticality = (heldout - baseline_eval) PEAK at intermediate recoverability
(edge states), and is that peak MEANINGFULLY > 0?

Hypothesis (criticality-weighted training, idea #2): most states are non-critical
(doomed p~0 or safe p~1 -> action can't change fate -> criticality 0); a rare
band of edge states (mid recoverability) is where the action decides the outcome.
If true, the +0.024 OVERALL realizable headroom is DILUTED by the many non-critical
states, and concentrating training on the edge band has a target.

NB: there is a MECHANICAL envelope — criticality is bounded and forced to 0 at
p=0 and p=1, so an inverted-U *shape* is trivial. The real question is the PEAK
VALUE at mid-p: >~0.08 => genuine action-sensitive (fate-deciding) states exist;
~0.02-0.04 => held-out removed it everywhere -> #2 (criticality training) dead too.

Run:
  cd oat && uv run python scripts/analyze_criticality.py -i my_datasets/iso_gate.npz
  (and -i my_datasets/iso_t20.npz)
"""
if __name__ == "__main__":
    import sys, os, pathlib
    sys.path.append(str(pathlib.Path(__file__).parent.parent))

import click
import numpy as np


@click.command()
@click.option('-i', '--input', 'inp', required=True, help='isolate .npz (with heldout/baseline_eval)')
@click.option('--xkey', default='baseline_eval', type=click.Choice(['baseline_eval', 'baseline']),
              help='recoverability proxy to bin by (x-axis)')
@click.option('--nbins', default=5, type=int)
def main(inp, xkey, nbins):
    d = np.load(inp, allow_pickle=True)
    if 'heldout' not in d.files:
        print(f"{inp} is not an isolate file (no 'heldout'). keys={d.files}")
        return
    held = d['heldout'].astype(float)
    base_e = d['baseline_eval'].astype(float) if 'baseline_eval' in d.files else d['baseline'].astype(float)
    base = d['baseline'].astype(float)
    crit = held - base_e                       # per-state realizable criticality
    x = (base_e if xkey == 'baseline_eval' else base)
    n = len(crit)

    print(f"\n=== inverted-U criticality test  ({inp}, n={n}) ===")
    print(f"overall criticality (heldout - baseline_eval) = {crit.mean():+.3f} "
          f"+/- {crit.std(ddof=1)/np.sqrt(n):.3f}")
    print(f"corr(criticality, recoverability[{xkey}]) = {np.corrcoef(crit, x)[0,1]:+.3f}\n")

    edges = np.linspace(0, 1, nbins + 1)
    print(f"  {'recov bin':>12} | {'n':>3} | {'mean recov':>10} | {'mean criticality':>16} | "
          f"{'envelope max':>12}")
    peak = (-1, -1.0)
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        m = (x >= lo) & (x < hi) if i < nbins - 1 else (x >= lo) & (x <= hi)
        if m.sum() == 0:
            print(f"  [{lo:.2f},{hi:.2f}) | {0:>3} | {'-':>10} | {'-':>16} | {'-':>12}")
            continue
        c = crit[m]
        mc = c.mean(); se = c.std(ddof=1)/np.sqrt(len(c)) if len(c) > 1 else float('nan')
        mr = x[m].mean()
        env = min(mr, 1 - mr)   # rough upper bound on achievable criticality at this recoverability
        print(f"  [{lo:.2f},{hi:.2f}) | {int(m.sum()):>3} | {mr:>10.3f} | "
              f"{mc:>+9.3f}+/-{se:<5.3f} | {env:>12.3f}")
        if mc > peak[1]:
            peak = (mr, mc)

    print(f"\nPEAK mean-criticality = {peak[1]:+.3f} at recoverability~{peak[0]:.2f}")
    print("READ: peak >~0.08 => fate-deciding (action-sensitive) edge states EXIST "
          "(overall +0.024 was diluted) -> criticality-weighted training has a target.")
    print("      peak ~0.02-0.05 => criticality is small even at the edge -> #2 dead too "
          "(held-out removed action-sensitivity everywhere).")
    print("\nDONE")


if __name__ == '__main__':
    main()
