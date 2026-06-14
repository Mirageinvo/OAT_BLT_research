"""
GATE A (criticality-gated compute) — offline existence test, NO sim.

Question: does the GENERATION-AWARE disagreement among the N sampled candidate plans
(action-space spread, available at inference WITHOUT sim) predict CRITICALITY
(realizable selection headroom = heldout - baseline_eval, which peaks at rare edge
states, recoverability ~ 0.5)? If yes, disagreement DETECTS edge states without obs
-> dodges the obs-wall that killed the K-predictor -> criticality-gated compute is
buildable (spend expensive compute only at edge). If disagreement is flat / uncorrelated
-> edge is not detectable from this cheap signal either -> proposal A is dead (fall back
to characterization, or RL which needs no per-state detection).

Needs an isolate npz produced with the updated branch_value_k.py (has `disagreement`).

Run:
  cd oat && uv run python scripts/gate_disagreement.py -i my_datasets/iso_gateA.npz
"""
if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).parent.parent))

import click
import numpy as np


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a):
    """Average ranks (1-based), tie-aware. Avoids a scipy dependency."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind='mergesort')
    sorted_a = a[order]
    ranks = np.empty(len(a), float)
    i, n = 0, len(a)
    while i < n:
        j = i
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0   # average rank over the tie block
        i = j
    return ranks


def auc(scores, labels):
    """ROC-AUC = P(score[pos] > score[neg]) via the Mann-Whitney U (rank-based, tie-aware).
    0.5 = no signal; >0.5 = higher scores mark positives; <0.5 = anti-signal."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    npos, nneg = int((labels == 1).sum()), int((labels == 0).sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    r = _rankdata(scores)
    U = r[labels == 1].sum() - npos * (npos + 1) / 2.0
    return float(U / (npos * nneg))


@click.command()
@click.option('-i', '--input', 'inp', required=True)
@click.option('--lo', default=0.35, type=float, help='edge band lower (recoverability)')
@click.option('--hi', default=0.65, type=float, help='edge band upper')
@click.option('--rkey', default='baseline', type=click.Choice(['baseline', 'baseline_eval']),
              help='recoverability axis (baseline = all-M mean, the unbiased one)')
def main(inp, lo, hi, rkey):
    d = np.load(inp, allow_pickle=True)
    if 'disagreement' not in d.files:
        print(f"no `disagreement` key — re-run branch_value_k.py --bon_isolate with the "
              f"updated script. keys={list(d.files)}"); return
    if 'heldout' not in d.files:
        print(f"not an isolate file (no heldout). keys={list(d.files)}"); return

    dis = d['disagreement'].astype(float)
    crit = d['heldout'].astype(float) - d['baseline_eval'].astype(float)
    recov = d[rkey].astype(float)
    n = len(dis)

    print(f"\n=== GATE A: disagreement -> criticality?  ({inp}, n={n}, rkey={rkey}) ===")
    print(f"disagreement: mean={dis.mean():.3f} std={dis.std():.3f} "
          f"[{dis.min():.3f}, {dis.max():.3f}]")

    # 1) does disagreement correlate with criticality (selection headroom)?
    print(f"\ncorr(disagreement, criticality)      = {pearson(dis, crit):+.3f}   "
          f"(>0 => high disagreement marks high-headroom states)")
    print(f"corr(disagreement, |criticality|)    = {pearson(dis, np.abs(crit)):+.3f}")
    print(f"corr(disagreement, recoverability)   = {pearson(dis, recov):+.3f}   "
          f"(edge ~ mid recov; expect inverted-U, so linear corr may be weak)")

    # 2) is disagreement ELEVATED in the edge band vs doomed/safe? (the detection test)
    doomed = recov < lo
    edge = (recov >= lo) & (recov <= hi)
    safe = recov > hi
    print(f"\ndisagreement by recoverability bin (edge should be HIGHEST if detectable):")
    for name, m in [('doomed(<%.2f)' % lo, doomed), ('EDGE[%.2f,%.2f]' % (lo, hi), edge),
                    ('safe(>%.2f)' % hi, safe)]:
        if m.sum() == 0:
            print(f"  {name:<16} n=0"); continue
        v = dis[m]
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan')
        c = crit[m]
        print(f"  {name:<16} n={int(m.sum()):3d}  disagreement={v.mean():.3f}±{se:.3f}   "
              f"(criticality here={c.mean():+.3f})")

    # 3) the decisive separation: edge vs doomed (edge is physically indistinguishable from
    #    doomed per characterize_edge -> can DISAGREEMENT separate them where physics can't?)
    if edge.sum() > 1 and doomed.sum() > 1:
        e, dm = dis[edge], dis[doomed]
        diff = e.mean() - dm.mean()
        se = np.sqrt(e.var(ddof=1) / len(e) + dm.var(ddof=1) / len(dm))
        z = diff / se if se > 0 else float('nan')
        print(f"\nEDGE vs DOOMED disagreement gap = {diff:+.3f}  (~{z:.1f} sigma)   "
              f"<- can disagreement separate edge from doomed where physics CANNOT?")

    # 4) AUC: disagreement as a detector. One number in [0.5,1] for 'detectability'.
    #    0.5 = no signal, >0.7 decent, >0.8 strong; <0.5 = disagreement is LOWER at edge.
    print("\nAUC of disagreement as an edge-detector (0.5=random, >0.7 decent, >0.8 strong):")
    if edge.sum() > 0 and (~edge).sum() > 0:
        print(f"  edge vs ALL-others (doomed+safe) = {auc(dis, edge.astype(int)):.3f}")
    if edge.sum() > 0 and doomed.sum() > 0:
        m = edge | doomed
        print(f"  edge vs DOOMED only (the hard case) = "
              f"{auc(dis[m], edge[m].astype(int)):.3f}   <- decisive: edge≈doomed physically")

    print("\nREAD:")
    print("  PASS (build A): disagreement ELEVATED at edge vs doomed AND safe (gap >0, >~2 sigma)")
    print("        and/or corr(disagreement, criticality) clearly >0. -> edge is detectable from")
    print("        a generation-aware signal (no obs) -> criticality-gated compute is buildable.")
    print("  FAIL (A dead): disagreement flat across bins / no corr -> edge not readable from this")
    print("        cheap signal either -> fall back to characterization paper, or RL (B, no detection).")
    print("\nDONE")


if __name__ == '__main__':
    main()
