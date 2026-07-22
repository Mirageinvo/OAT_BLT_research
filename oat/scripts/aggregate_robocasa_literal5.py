#!/usr/bin/env python3
"""Aggregate RoboCasa literal-5-seed Wave1/2 evals (ROBOCASA.md §4).

Reads:
  <root>/{baseline,bon_n8,awr}_seed{10000..10004}/eval_log.json

Writes:
  <root>/summary_literal5.json

Formulas (n=5 seeds):
  mean, SD (sample), SEM = SD/sqrt(n)
  Δ = mean(method) - mean(baseline)
  SEM_Δ = sqrt(SEM_m^2 + SEM_b^2)   # unpaired / conservative (locked in ROBOCASA.md §4)
                                      # paired SEM of per-seed diffs is optional later; not used here
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

SEEDS = (10000, 10001, 10002, 10003, 10004)
METHODS = ("baseline", "bon_n8", "awr")


def _sr_from_eval_log(path: pathlib.Path) -> float:
    """Read SR from eval_policy_sim.py dump.

    For any -n (including -n 1), eval_policy_sim writes ``{key}_mean`` from the
    runner's ``mean_success_rate`` / ``<task>/mean_success_rate``. So paper
    literal-5 logs use ``mean_success_rate_mean`` (and optionally nested
    ``*/mean_success_rate_mean``). Bare ``mean_success_rate`` is a safety fallback
    only (raw runner dict / hand-copied logs).
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    if "mean_success_rate_mean" in d:
        return float(d["mean_success_rate_mean"])
    # nested "<task>/mean_success_rate_mean"
    keys = [k for k in d if str(k).endswith("mean_success_rate_mean")]
    if keys:
        return float(d[keys[0]])
    if "mean_success_rate" in d:
        return float(d["mean_success_rate"])
    raise KeyError(f"no success rate in {path}; keys={list(d)[:20]}")


def _stats(vals: List[float]) -> Dict[str, float]:
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        sd = 0.0
    else:
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    sem = sd / math.sqrt(n) if n else 0.0
    return {"mean": mean, "sd": sd, "sem": sem, "n": float(n)}


def _collect(root: pathlib.Path, method: str) -> Tuple[Dict[str, float], List[pathlib.Path]]:
    per_seed: Dict[str, float] = {}
    paths: List[pathlib.Path] = []
    for seed in SEEDS:
        p = root / f"{method}_seed{seed}" / "eval_log.json"
        if not p.is_file():
            continue
        per_seed[str(seed)] = _sr_from_eval_log(p)
        paths.append(p)
    return per_seed, paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, required=True)
    args = ap.parse_args()
    root = args.root
    if not root.is_dir():
        print(f"ERROR: missing root {root}", file=sys.stderr)
        raise SystemExit(1)

    out: Dict[str, Any] = {
        "protocol": {
            "layout": "literal_5_seeds",
            "seeds": list(SEEDS),
            "n_test": 50,
            "n_exp_per_seed": 1,
            "aggregation": "mean over seeds; primary uncertainty SEM; also SD",
            "delta": "mean(method)-mean(baseline); SEM_delta=sqrt(SEM_m^2+SEM_b^2) unpaired/conservative (paired SEM of diffs optional later)",
        },
        "methods": {},
        "deltas": {},
    }

    baseline_stats: Optional[Dict[str, float]] = None
    for method in METHODS:
        per_seed, paths = _collect(root, method)
        if len(per_seed) == 0:
            continue
        if len(per_seed) != len(SEEDS):
            print(
                f"WARN {method}: found {len(per_seed)}/{len(SEEDS)} seeds: {sorted(per_seed)}",
                file=sys.stderr,
            )
        vals = [per_seed[str(s)] for s in SEEDS if str(s) in per_seed]
        st = _stats(vals)
        out["methods"][method] = {
            "per_seed_sr": per_seed,
            "mean": st["mean"],
            "sd": st["sd"],
            "sem": st["sem"],
            "n_seeds": int(st["n"]),
            "eval_logs": [str(p) for p in paths],
        }
        if method == "baseline":
            baseline_stats = st
        print(
            f"{method}: mean={st['mean']:.4f}  SEM={st['sem']:.4f}  SD={st['sd']:.4f}  "
            f"seeds={per_seed}"
        )

    if baseline_stats is not None:
        for method in ("bon_n8", "awr"):
            if method not in out["methods"]:
                continue
            m = out["methods"][method]
            delta = m["mean"] - baseline_stats["mean"]
            sem_d = math.sqrt(m["sem"] ** 2 + baseline_stats["sem"] ** 2)
            out["deltas"][f"delta_{method}_minus_baseline"] = {
                "delta": delta,
                "sem_delta": sem_d,
            }
            print(f"Δ({method}-baseline)={delta:+.4f}  SEM_Δ={sem_d:.4f}")

    dest = root / "summary_literal5.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
