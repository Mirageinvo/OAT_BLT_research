#!/usr/bin/env python3
"""Aggregation for the AAMAS-27 rebuttal CS ablation (LIBERO-LONG).

  # bandwidth calibration (median of adaptive prefix h over all logged decisions)
  python scripts/rebuttal/aggregate_ablation.py calib RUN_DIR

  # SR table, paired deltas vs cs_prefix, disagreement, timing
  python scripts/rebuttal/aggregate_ablation.py ablation DIR_cs_prefix DIR_cs_endpoint ... \
      [--control cs_prefix] [--n_boot 10000] [--out rebuttal_results_auto.md]

Each DIR must contain episodes.csv written by eval_policy_sim.py (and optionally
selector_decisions.csv). Only numpy/pandas needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def _md(df, **kw):
    try:
        return df.to_markdown(**kw)
    except ImportError:                     # tabulate not installed
        return "```\n" + df.to_string(index=kw.get('index', True)) + "\n```"


def _load(dirs, name):
    frames = []
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cmd_calib(a):
    dec = _load([a.run_dir], 'selector_decisions.csv')
    if dec.empty:
        sys.exit("no selector_decisions.csv (run with --log_diagnostics)")
    h = dec['h_prefix'].to_numpy()
    out = dict(
        source_dir=os.path.abspath(a.run_dir),
        n_decisions=int(len(h)), n_episodes=int(dec.groupby(['init_id', 'repeat_seed'], dropna=False).ngroups),
        seed_labels=[int(dec['seed_label'].min()), int(dec['seed_label'].max())],
        h_median=float(np.median(h)),
        h_q10_q90=[float(np.quantile(h, .1)), float(np.quantile(h, .9))],
        h_mean=float(h.mean()),
    )
    print(json.dumps(out, indent=2))
    with open(os.path.join(a.run_dir, 'h_calibration.json'), 'w') as f:
        json.dump(out, f, indent=2)


def _paired_boot(diff_by_scene: pd.DataFrame, n_boot: int, seed: int = 0):
    """diff_by_scene: columns task, d (per-scene diff already averaged over repeats).
    Resample scenes WITHIN each task, equal task weights. Returns (point, lo, hi) in pp."""
    rng = np.random.default_rng(seed)
    groups = [g['d'].to_numpy() for _, g in diff_by_scene.groupby('task')]
    point = np.mean([g.mean() for g in groups])
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = np.mean([g[rng.integers(0, len(g), len(g))].mean() for g in groups])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return 100 * point, 100 * lo, 100 * hi


def cmd_ablation(a):
    # every DIR must hold exactly one variant's episodes.csv; nothing is skipped silently
    pre = []
    frames = []
    for d in a.dirs:
        p = os.path.join(d, 'episodes.csv')
        if not os.path.isfile(p):
            pre.append(f"missing {p}")
            continue
        f = pd.read_csv(p)
        if f.empty:
            pre.append(f"empty {p}")
            continue
        vs = sorted(f['variant'].astype(str).unique())
        if len(vs) != 1:
            pre.append(f"{p} holds {len(vs)} variants {vs} (expected exactly 1)")
        frames.append(f)
    ep = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not ep.empty:
        per_dir = [sorted(f['variant'].astype(str).unique()) for f in frames]
        flat = [v for vs in per_dir for v in vs]
        if len(flat) != len(set(flat)):
            pre.append(f"same variant in several DIRs: {flat}")
        if a.control not in set(flat):
            pre.append(f"control `{a.control}` not among loaded variants {sorted(set(flat))}")
        if len(set(flat)) < 2:
            pre.append(f"need >= 2 variants for a paired comparison, got {sorted(set(flat))}")
    if pre and not a.no_strict:
        sys.exit("INPUT FAILURE — paired statistics not computed:\n  - " + "\n  - ".join(pre))
    if ep.empty:
        sys.exit("no episodes.csv found")
    ep['repeat_seed'] = ep['repeat_seed'].fillna(-1).astype(int)
    lines = []
    P = lines.append

    # ---- integrity (strict: paired statistics are refused if pairing is not exact) ----
    P("## Integrity\n")
    errors = list(pre)          # only non-empty under --no_strict
    key = ['task', 'init_id', 'repeat_seed']
    variants = list(dict.fromkeys(ep['variant']))
    bad = ep[ep['status'] != 'ok']
    P(f"- episodes: {len(ep)}; variants: {variants}")
    P(f"- non-ok status rows: {len(bad)} {dict(bad['status'].value_counts()) if len(bad) else ''}")
    if len(bad):
        errors.append(f"{len(bad)} rows with status != ok")
    unfinished = int((ep['episode_finished'] == 0).sum())
    P(f"- unfinished episodes (loop cap without done): {unfinished}")
    if unfinished:
        errors.append(f"{unfinished} unfinished episodes")
    dup = int(ep.duplicated(['variant'] + key).sum())
    P(f"- duplicate (variant, task, init_id, repeat_seed) rows: {dup}")
    if dup:
        errors.append(f"{dup} duplicate rows")
    h = ep['scene_hash'].astype(str)
    invalid_h = ep[ep['scene_hash'].isna() | h.str.startswith('err:') | (h.str.len() != 16)]
    P(f"- rows with missing/invalid scene_hash: {len(invalid_h)}")
    if len(invalid_h):
        errors.append(f"{len(invalid_h)} rows with invalid scene_hash")
    keysets = {v: set(map(tuple, ep.loc[ep['variant'] == v, key].itertuples(index=False)))
               for v in variants}
    ref = keysets[variants[0]]
    for v in variants[1:]:
        if keysets[v] != ref:
            msg = (f"key set of `{v}` != `{variants[0]}` "
                   f"(only in {variants[0]}: {len(ref - keysets[v])}, only in {v}: {len(keysets[v] - ref)})")
            P(f"- {msg}")
            errors.append(msg)
    hk = ep.groupby(key)['scene_hash'].nunique()
    n_hk = int((hk > 1).sum())
    P(f"- (task, init_id, repeat_seed) keys whose scene hash differs across variants: {n_hk} / {len(hk)}")
    if n_hk:
        errors.append(f"{n_hk} keys with mismatching scene hashes across variants")
    hr = ep.groupby(['task', 'init_id'])['scene_hash'].nunique()
    P(f"- (task, init_id) whose hash differs across repeat seeds (expected 0 with seeded resets): "
      f"{int((hr > 1).sum())} / {len(hr)}")
    tk = ep.groupby('init_id')['task'].nunique()
    P(f"- init_ids mapped to >1 task: {int((tk > 1).sum())}")
    if 'k_tokens' in ep and 'r_exec' in ep:
        P("- generated tokens K / executed chunk length R per variant (values seen in any episode):")
        for v in variants:
            e = ep[ep['variant'] == v]
            ks = sorted({x for s_ in e['k_tokens'].astype(str) for x in s_.split('|')})
            rs = sorted({x for s_ in e['r_exec'].astype(str) for x in s_.split('|')})
            P(f"  - `{v}`: K={ks}, R={rs}")
            if a.expect_k is not None and ks != [str(a.expect_k)]:
                errors.append(f"`{v}`: K values {ks} != [{a.expect_k}]")
            if a.expect_r is not None and rs != [str(a.expect_r)]:
                errors.append(f"`{v}`: R values {rs} != [{a.expect_r}]")
    dec = _load(a.dirs, 'selector_decisions.csv')
    if not dec.empty:
        own = {'cs_prefix': 'cs_prefix_idx', 'medoid_prefix': 'medoid_prefix_idx',
               'cs_endpoint': 'cs_endpoint_idx', 'cs_prefix_fixed': 'cs_prefix_fixed_idx'}
        P("- executed index vs the variant's own shadow decision (must be 0 mismatches):")
        for v, g in dec.groupby('variant'):
            if v not in own:
                continue
            mm = int((g['executed_idx'] != g[own[v]]).sum())
            Rs = sorted(g['R'].unique().tolist()) if 'R' in g else 'n/a'
            P(f"  - `{v}`: {mm} / {len(g)} mismatches; decision R values: {Rs}")
            if mm:
                errors.append(f"`{v}`: {mm} executed-index mismatches vs own diagnostics")
            if a.expect_r is not None and 'R' in g and Rs != [a.expect_r]:
                errors.append(f"`{v}`: decision R values {Rs} != [{a.expect_r}]")
    if not ('k_tokens' in ep and 'r_exec' in ep) and (a.expect_k is not None or a.expect_r is not None):
        errors.append("episodes.csv has no k_tokens/r_exec columns (pre-08a1205 runner) — K/R unverifiable")
    cnt = ep.groupby(['variant', 'repeat_seed']).size().unstack(fill_value=0)
    P("\nEpisodes per variant × repeat_seed:\n")
    P(_md(cnt, ))
    P("")
    if errors and not a.no_strict:
        txt = "\n".join(lines)
        print(txt)
        sys.exit("INTEGRITY FAILURE — paired statistics not computed:\n  - " + "\n  - ".join(errors)
                 + "\n(--no_strict prints the tables anyway; never report numbers from such a run)")
    elif errors:
        P("**WARNING: integrity failures above; numbers below are NOT valid paired comparisons.**\n")

    # ---- SR ----
    P("## Success rate\n")
    rep_sr = ep.groupby(['variant', 'repeat_seed'])['success'].mean()
    rows = []
    for v in variants:
        r = rep_sr[v]
        rows.append(dict(variant=v, repeats=len(r), episodes=int((ep['variant'] == v).sum()),
                         SR=100 * r.mean(), std_over_repeats=(100 * r.std(ddof=1) if len(r) > 1 else np.nan)))
    P(_md(pd.DataFrame(rows), index=False, floatfmt='.1f'))
    P("\nPer task (SR %, pooled over repeats):\n")
    pt = (100 * ep.groupby(['task', 'variant'])['success'].mean()).unstack()[variants]
    pt.index = [t[:60] for t in pt.index]
    P(_md(pt, floatfmt='.1f'))
    P("")

    # ---- paired deltas ----
    ctrl = a.control
    if ctrl in variants:
        P(f"## Paired Δ vs `{ctrl}` (pp; 95% bootstrap CI: scenes resampled within task, "
          f"equal task weight; per-scene diffs averaged over shared repeats)\n")
        base = ep[ep['variant'] == ctrl].set_index(['init_id', 'repeat_seed'])
        rows = []
        for v in variants:
            if v == ctrl:
                continue
            other = ep[ep['variant'] == v].set_index(['init_id', 'repeat_seed'])
            j = base[['task', 'success']].join(other[['success']].rename(columns={'success': 'success_v'}), how='inner')
            j['d'] = j['success_v'] - j['success']
            per_scene = j.reset_index().groupby(['task', 'init_id'])['d'].mean().reset_index()
            pt_, lo, hi = _paired_boot(per_scene, a.n_boot)
            disc = j.reset_index()
            rows.append(dict(variant=v, paired_episodes=len(j), scenes=len(per_scene),
                             delta_pp=pt_, ci_lo=lo, ci_hi=hi,
                             v_only_success=int(((disc.success_v == 1) & (disc.success == 0)).sum()),
                             ctrl_only_success=int(((disc.success_v == 0) & (disc.success == 1)).sum())))
        P(_md(pd.DataFrame(rows), index=False, floatfmt='.1f'))
        P("")

    # ---- disagreement (same candidates, on control trajectories) ----
    if not dec.empty:
        dc = dec[dec['variant'] == ctrl] if ctrl in set(dec['variant']) else dec
        P(f"## Selector disagreement on the SAME 8 candidates (decisions on `{dc['variant'].iloc[0]}` trajectories)\n")
        dc = dc.assign(idx_cs_med=(dc.cs_prefix_idx != dc.medoid_prefix_idx),
                       pre_cs_med=(dc.d_cs_medoid > 0),
                       idx_cs_ep=(dc.cs_prefix_idx != dc.cs_endpoint_idx),
                       pre_cs_ep=(dc.d_cs_endpoint > 0))
        cols = ['idx_cs_med', 'pre_cs_med', 'idx_cs_ep', 'pre_cs_ep']
        if (dc.cs_prefix_fixed_idx >= 0).any():
            dc = dc.assign(idx_cs_fixed=(dc.cs_prefix_idx != dc.cs_prefix_fixed_idx))
            cols.append('idx_cs_fixed')
        tot = (100 * dc[cols].mean()).to_frame('all_decisions_%').T
        tot.insert(0, 'decisions', len(dc))
        P(_md(tot, index=False, floatfmt='.1f'))
        P("\n`idx_*` = different selected index; `pre_*` = different executed prefix "
          "(normalized L2 > 0; duplicates/ties do not count).\n")
        bt = 100 * dc.groupby('task')[cols].mean()
        bt.index = [t[:60] for t in bt.index]
        P(_md(bt, floatfmt='.1f'))
        m = dc[dc.pre_cs_med]
        if len(m):
            P(f"\nWhen CS and medoid prefixes differ: median normalized distance "
              f"{m.d_cs_medoid.median():.3f} (h median {m.h_prefix.median():.3f}); "
              f"episodes with ≥1 such decision: "
              f"{m.groupby(['init_id', 'repeat_seed']).ngroups}/{dc.groupby(['init_id', 'repeat_seed']).ngroups}.")
        P("")

    # ---- timing ----
    P("## Inference cost (as logged; batched calls, see batch_envs)\n")
    rows = []
    for v in variants:
        e = ep[ep['variant'] == v]
        rows.append(dict(variant=v, SR=100 * e.success.mean(), calls_per_ep=e.policy_calls.mean(),
                         ms_per_ep=e.inference_ms.mean(),
                         ms_per_step=e.inference_ms.sum() / max(e.env_steps.sum(), 1),
                         ms_per_call=e.inference_ms.sum() / max(e.policy_calls.sum(), 1),
                         batch_envs=int(e.batch_envs.iloc[0])))
    P(_md(pd.DataFrame(rows), index=False, floatfmt='.2f'))
    P("\nms values are wall-clock per batched policy call (all envs of the batch share one call); "
      "compare only rows with the same batch_envs and GPU.")

    txt = "\n".join(lines)
    print(txt)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('calib')
    c.add_argument('run_dir')
    b = sub.add_parser('ablation')
    b.add_argument('dirs', nargs='+')
    b.add_argument('--control', default='cs_prefix')
    b.add_argument('--n_boot', type=int, default=10000)
    b.add_argument('--out', default=None)
    b.add_argument('--expect_k', type=int, default=8, help='required generated-token count (strict)')
    b.add_argument('--expect_r', type=int, default=16, help='required executed chunk length (strict)')
    b.add_argument('--no_strict', action='store_true',
                   help='print tables even if pairing integrity fails (diagnostics only)')
    a = ap.parse_args()
    {'calib': cmd_calib, 'ablation': cmd_ablation}[a.cmd](a)


if __name__ == '__main__':
    main()
