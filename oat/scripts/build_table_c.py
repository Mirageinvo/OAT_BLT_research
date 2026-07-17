#!/usr/bin/env python3
"""Build output/eval/matched_s10000/table_c.json from per-suite latency.json."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MATCHED = ROOT / "output/eval/matched_s10000"
SUITES = ["can", "coffee-pull", "stick-pull", "disassemble", "box-close", "square"]


def sr_pct(block):
    if not isinstance(block, dict):
        return None
    if block.get("pct"):
        return block["pct"]
    if "mean" in block:
        return f"{100 * block['mean']:.1f} ± {100 * block.get('std', 0):.1f}%"
    return None


def main():
    # backfill iqr into existing latency.json
    for p in MATCHED.glob("*/latency.json"):
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
        p = MATCHED / s / "latency.json"
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        modes = d["modes"]
        sr = d.get("table_p_sr") or {}

        def ms(m):
            x = modes[m]
            return {k: x.get(k) for k in ("median_ms", "mean_ms", "std_ms", "iqr_ms", "reps")}

        rows[s] = {
            "latency": {"single": ms("single"), "bon": ms("bon"), "awr": ms("awr")},
            "sr_table_p": {
                "baseline": sr_pct(sr.get("baseline")),
                "bon": sr_pct(sr.get("bon")),
                "awr": sr_pct(sr.get("awr")),
            },
            "latency_json": str(p.relative_to(ROOT)),
            "paper_proof": d.get("paper_proof", False),
            "git_commit": d.get("git_commit"),
            "git_source": d.get("git_source"),
            "gpu_name": d.get("gpu_name"),
            "torch_version": d.get("torch_version"),
            "cuda_version": d.get("cuda_version"),
            "measured_at": d.get("measured_at"),
            "base_ckpt": d.get("base_ckpt"),
            "awr_ckpt": d.get("awr_ckpt"),
        }

    out = {
        "protocol": "RESOLUTIONPLAN Latency/Table C paper-proof 2026-07-17",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "policy-forward ms only; SR from Table P. "
            "BoN often ≈ single when vision dominates (amortized encode once) — expected on OAT. "
            "paper_proof requires non-empty git_commit + per-mode obs reset."
        ),
        "suites": rows,
    }
    out_path = MATCHED / "table_c.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path} n={len(rows)}")
    for s, r in rows.items():
        L = r["latency"]
        print(
            f"  {s:12} single={L['single']['median_ms']:.1f} "
            f"bon={L['bon']['median_ms']:.1f} awr={L['awr']['median_ms']:.1f}"
        )


if __name__ == "__main__":
    main()
