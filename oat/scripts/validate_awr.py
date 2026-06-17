"""
Validate an AWR/ReST dataset (collect_awr_dataset.py output) BEFORE training. Checks task
coverage (the single-task bug), shapes/dtypes, NaN/inf, token validity, success/advantage
signal usability, episode-phase coverage, and previews the AWR weights at a given beta.

Run:
  cd oat && uv run python scripts/validate_awr.py -i my_datasets/awr_bon.npz --beta 0.5
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import collections

import click
import numpy as np


@click.command()
@click.option('-i', '--input', 'inp', default='my_datasets/awr_bon.npz')
@click.option('--beta', default=0.5, type=float, help='AWR temperature for the weight preview')
@click.option('--w_max', default=20.0, type=float)
def main(inp, beta, w_max):
    d = np.load(inp, allow_pickle=True)
    feats = d['features']; toks = d['tokens']; succ = d['success']
    ep = d['ep_id']; step = d['step']; task = d['task'].astype(str)
    N = len(feats)
    flags = []
    print(f"=== validate {inp}  (N={N}) ===\n")

    # 1) shapes / dtypes
    print(f"[shapes] features {feats.shape} {feats.dtype} | tokens {toks.shape} {toks.dtype} | "
          f"success {succ.shape} | ep_id {ep.shape} | step {step.shape}")
    if feats.ndim != 3:
        flags.append("features not 3-D [N,To,d]")

    # 2) NaN / inf in features
    nbad = int(np.isnan(feats).sum() + np.isinf(feats).sum())
    print(f"[finite] non-finite feature entries = {nbad}  "
          f"(feat range [{np.nanmin(feats):.2f}, {np.nanmax(feats):.2f}])")
    if nbad:
        flags.append(f"{nbad} non-finite feature entries")

    # 3) token validity
    print(f"[tokens] value range [{toks.min()}, {toks.max()}]  "
          f"uniq first-token = {len(np.unique(toks[:, 0]))}  uniq full-seq = {len(np.unique(toks, axis=0))}")
    if toks.min() < 0:
        flags.append("negative token id")
    if len(np.unique(toks, axis=0)) < 0.1 * N:
        flags.append("token sequences highly degenerate (<10% unique)")

    # 4) task coverage (the single-task bug)
    c = collections.Counter(task)
    print(f"\n[coverage] {len(c)} unique tasks")
    sr_by_task = {}
    for t, cnt in sorted(c.items(), key=lambda x: -x[1]):
        m = task == t
        sr_by_task[t] = succ[m].mean()
        print(f"   {cnt:6d} ch  SR={succ[m].mean():.3f}  {t[:62]}")
    if len(c) < 2:
        flags.append("SINGLE-TASK dataset (collection bug not fixed!)")
    elif len(c) < 10:
        flags.append(f"only {len(c)} tasks (<10 libero10 subtasks)")
    bal = max(c.values()) / max(1, min(c.values()))
    print(f"   balance max/min chunks = {bal:.1f}x")

    # 5) success / advantage usability
    n_ep = len(np.unique(ep))
    ep_sr = np.array([succ[ep == e][0] for e in np.unique(ep)])
    print(f"\n[success] chunk-weighted SR (AWR baseline) = {succ.mean():.3f} | "
          f"per-episode SR = {ep_sr.mean():.3f} over {n_ep} episodes")
    print(f"          succ chunks: pos={int(succ.sum())} ({succ.mean()*100:.1f}%)  "
          f"neg={int((1-succ).sum())} ({(1-succ.mean())*100:.1f}%)")
    if succ.mean() < 0.05 or succ.mean() > 0.95:
        flags.append(f"degenerate success rate {succ.mean():.3f} -> no advantage contrast")

    # 6) episode-phase coverage
    print(f"[phase]   step range [{step.min()}, {step.max()}]  "
          f"median {int(np.median(step))}  p90 {int(np.percentile(step,90))}")

    # 7) AWR weight preview
    baseline = succ.mean()
    w = np.exp((succ - baseline) / beta).clip(max=w_max)
    w = w / w.mean()
    print(f"\n[awr@beta={beta}] success:fail raw-weight ratio = exp(1/beta) = {np.exp(1/beta):.1f}x "
          f"| normalized weight min/mean/max = {w.min():.2f}/{w.mean():.2f}/{w.max():.2f}")

    # verdict
    print("\n=== VERDICT ===")
    if not flags:
        print("PASS — dataset looks healthy for AWR training.")
    else:
        print("CONCERNS:")
        for f in flags:
            print(f"  - {f}")
    print(f"\nExpectation: per-episode SR ~0.69 (multi-task BoN-distill) confirms a strong "
          f"source; ~0.58 = base-level (BoN gain not captured); check per-task SR for outliers.")


if __name__ == '__main__':
    main()
