#!/usr/bin/env python3
"""Build table_c.json (or table_c_fair_kv.json) from per-suite latency artifacts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MATCHED = ROOT / "output/eval/matched_s10000"
SUITES = [
    "can",
    "coffee-pull",
    "stick-pull",
    "disassemble",
    "box-close",
    "square",
    "lift",
]


def sr_pct(block):
    if not isinstance(block, dict):
        return None
    if block.get("pct"):
        return block["pct"]
    if "mean" in block:
        return f"{100 * block['mean']:.1f} ± {100 * block.get('std', 0):.1f}%"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fair_kv",
        action="store_true",
        help="aggregate latency_fair_kv.json → table_c_fair_kv.json (rebuttal)",
    )
    args = ap.parse_args()
    fname = "latency_fair_kv.json" if args.fair_kv else "latency.json"
    out_name = "table_c_fair_kv.json" if args.fair_kv else "table_c.json"

    # backfill iqr
    for p in MATCHED.glob(f"*/{fname}"):
        d = json.loads(p.read_text())
        changed = False
        for mode in d.get("modes", {}).values():
            if "iqr_ms" not in mode and "samples_ms" in mode:
                arr = np.asarray(mode["samples_ms"], dtype=float)
                q25, q75 = np.percentile(arr, [25, 75])
                mode["p25_ms"] = float(q25)
                mode["p75_ms"] = float(q75)
                mode["iqr_ms"] = float(q75 - q25)
                changed = True
        if changed:
            p.write_text(json.dumps(d, indent=2) + "\n")
            print("patched", p)

    rows = {}
    for s in SUITES:
        p = MATCHED / s / fname
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        modes = d["modes"]
        sr = d.get("table_p_sr") or {}

        def ms(m):
            x = modes[m]
            keys = (
                "median_ms",
                "mean_ms",
                "std_ms",
                "sem_ms",
                "iqr_ms",
                "reps",
                "n_trials",
                "trial_median_mean_ms",
                "trial_median_std_ms",
            )
            return {k: x.get(k) for k in keys if k in x or k in ("median_ms", "std_ms", "reps")}

        rows[s] = {
            "latency": {"single": ms("single"), "bon": ms("bon"), "awr": ms("awr")},
            "sr_table_p": {
                "baseline": sr_pct(sr.get("baseline")),
                "bon": sr_pct(sr.get("bon")),
                "awr": sr_pct(sr.get("awr")),
            },
            "latency_json": str(p.relative_to(ROOT)),
            "paper_proof": d.get("paper_proof", False),
            "fair_kv": d.get("fair_kv", False),
            "batch_size": d.get("batch_size", 1),
            "trials": d.get("trials"),
            "git_commit": d.get("git_commit"),
            "git_source": d.get("git_source"),
            "gpu_name": d.get("gpu_name"),
            "torch_version": d.get("torch_version"),
            "cuda_version": d.get("cuda_version"),
            "measured_at": d.get("measured_at"),
            "base_ckpt": d.get("base_ckpt"),
            "awr_ckpt": d.get("awr_ckpt"),
            "bon_minus_single_ms": modes["bon"]["median_ms"] - modes["single"]["median_ms"],
        }

    out = {
        "protocol": (
            "RESOLUTIONPLAN Latency fair-KV 2026-07-18"
            if args.fair_kv
            else "RESOLUTIONPLAN Latency/Table C paper-proof 2026-07-17"
        ),
        "fair_kv": args.fair_kv,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "FAIR_KV rebuttal: Single/AWR = predict_action (KV-cache); BoN = generate (KV). "
            "Does not replace Table C deployed paths."
            if args.fair_kv
            else (
                "policy-forward ms only; SR from Table P. "
                "BoN often ≈ single when vision dominates (amortized encode once). "
                "paper_proof requires non-empty git_commit + per-mode obs reset."
            )
        ),
        "suites": rows,
    }
    out_path = MATCHED / out_name
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path} n={len(rows)}")
    for s, r in rows.items():
        L = r["latency"]
        dbs = r["bon_minus_single_ms"]
        def fmt(x):
            std = x.get("std_ms")
            if std is None:
                return f"{x['median_ms']:.1f}"
            return f"{x['median_ms']:.1f}±{std:.1f}"

        print(
            f"  {s:12} single={fmt(L['single'])} "
            f"bon={fmt(L['bon'])} awr={fmt(L['awr'])} "
            f"bon-single={dbs:+.2f}ms trials={r.get('trials')}"
        )


if __name__ == "__main__":
    main()
