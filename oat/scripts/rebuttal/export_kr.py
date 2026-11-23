"""Export the single-deviation K/R oracle data (Table 2 source) to CSV + a summary of the
per-state distribution behind the reported means.

Input: npz from branch_value_k.py grid mode with keys
  task, step, grip_will_change, ncon, eef_vel, gripper_open, recon_gap,
  p_kc_rs, p_kc_rl, p_kf_rs, p_kf_rl     (success fractions over M continuations per branch)
kc/kf = coarse/full token depth, rs/rl = short/long executed horizon.

Only per-branch success fractions are stored (not individual continuation outcomes), so the CSV
holds success COUNTS out of M per branch. The script fails if p*M is not an integer.

  python scripts/rebuttal/export_kr.py -i oracle_grid_mt.npz -o rebuttal/kr_continuations.csv \
      --summary rebuttal/kr_summary.md --M 5 --k_coarse 1 --k_full 8 --r_short 8 --r_long 32
"""
import argparse
from math import comb

import numpy as np
import pandas as pd


def sign_test(n_pos, n_neg):
    n, k = n_pos + n_neg, min(n_pos, n_neg)
    if n == 0:
        return float("nan")
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--out_csv", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--M", type=int, required=True, help="continuations per branch")
    ap.add_argument("--k_coarse", type=int, required=True)
    ap.add_argument("--k_full", type=int, required=True)
    ap.add_argument("--r_short", type=int, required=True)
    ap.add_argument("--r_long", type=int, required=True)
    ap.add_argument("--n_boot", type=int, default=10000)
    a = ap.parse_args()

    d = np.load(a.input, allow_pickle=True)
    M, kc, kf, rs, rl = a.M, a.k_coarse, a.k_full, a.r_short, a.r_long
    branches = {"p_kc_rs": (kc, rs), "p_kc_rl": (kc, rl), "p_kf_rs": (kf, rs), "p_kf_rl": (kf, rl)}
    n = len(d["p_kf_rl"])
    rows = {"state_id": np.arange(n)}
    for key in ["task", "step", "grip_will_change", "ncon", "eef_vel", "gripper_open", "recon_gap"]:
        rows[key] = d[key]
    cnt = {}
    for key, (k, r) in branches.items():
        x = np.asarray(d[key], dtype=float) * M
        if not np.allclose(x, np.round(x), atol=1e-6):
            raise SystemExit(f"{key}*M is not integer (M={M}?) -> refusing to export counts")
        cnt[key] = np.round(x).astype(int)
        rows[f"succ_K{k}_R{r}"] = cnt[key]
    rows["M"] = M
    pd.DataFrame(rows).to_csv(a.out_csv, index=False)

    tasks = np.asarray(d["task"])
    rng = np.random.default_rng(0)

    def boot_ci(diff):  # resample states within task
        groups = [np.where(tasks == t)[0] for t in np.unique(tasks)]
        ms = []
        for _ in range(a.n_boot):
            idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
            ms.append(diff[idx].mean())
        return np.percentile(ms, [2.5, 97.5])

    # (label, plus branch, minus branch, reported in Table 2?)
    contrasts = [
        (f"K {kc}->{kf} at R={rl}", "p_kf_rl", "p_kc_rl", True),
        (f"R {rl}->{rs} at K={kf}", "p_kf_rs", "p_kf_rl", True),
        (f"K {kc}->{kf} at R={rs}", "p_kf_rs", "p_kc_rs", False),
        (f"R {rl}->{rs} at K={kc}", "p_kc_rs", "p_kc_rl", False),
    ]
    L = [f"# Single-deviation K/R oracle: per-state distribution",
         f"", f"Source: `{a.input}`; n={n} states, {len(np.unique(tasks))} tasks; M={M} continuations per branch.",
         f"Effect per state = (successes_plus - successes_minus)/M (one outcome = {100 / M:.0f} p.p.).",
         f"Ties split by level: both 0/M (no success in either branch), both M/M, both in between.",
         f"Sign test = exact two-sided binomial on non-tied states. CI = 95% bootstrap over states "
         f"within task ({a.n_boot} resamples, seed 0). Observed differences include continuation noise.", "",
         "| contrast | Table 2 | mean | 95% CI | improved | unchanged (0/0, M/M, mid) | worse | sign-test p |",
         "|---|---|---|---|---|---|---|---|"]
    for lab, plus, minus, rep in contrasts:
        diff = (cnt[plus] - cnt[minus]) / M
        pos, neg = int((diff > 0).sum()), int((diff < 0).sum())
        tie = diff == 0
        t0 = int((tie & (cnt[plus] == 0)).sum())
        tM = int((tie & (cnt[plus] == M)).sum())
        tmid = int(tie.sum()) - t0 - tM
        lo, hi = boot_ci(diff)
        L.append(f"| {lab} | {'yes' if rep else 'supplementary'} | {diff.mean():+.4f} | [{lo:+.3f}, {hi:+.3f}] | "
                 f"{pos} ({100 * pos / n:.1f}%) | {int(tie.sum())} ({t0}, {tM}, {tmid}) | {neg} ({100 * neg / n:.1f}%) | "
                 f"{sign_test(pos, neg):.3f} |")
    k_eff = (cnt["p_kf_rl"] - cnt["p_kc_rl"]) / M
    L += ["", f"corr(K effect at R={rl}, recon_gap) = {np.corrcoef(k_eff, np.asarray(d['recon_gap'], float))[0, 1]:+.4f}",
          "", "Branch success counts (states with c successes out of M):", "",
          "| branch | " + " | ".join(str(c) for c in range(M + 1)) + " | mean p |",
          "|---|" + "---|" * (M + 2)]
    for key, (k, r) in branches.items():
        h = np.bincount(cnt[key], minlength=M + 1)
        L.append(f"| K={k}, R={r} | " + " | ".join(str(x) for x in h) + f" | {cnt[key].mean() / M:.3f} |")
    open(a.summary, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
