#!/usr/bin/env python3
"""Selector-only latency (plan sec. 13): vote (CS) / medoid / kdpe / random on ready candidate
tensors [N,H,7]. No policy forward, no checkpoint. Complements measure_latency_paper.py
(--bon_signal kdpe) which times the FULL policy path.

  python scripts/measure_selector_latency.py -d cpu
  python scripts/measure_selector_latency.py -d cuda:0 --N 8 --R 16 --out latency_selector_only.json
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import click
import torch

from oat.policy.oatpolicy import OATPolicy


class _IdNorm:
    """Stand-in for the OAT normalizer: same op cost class (affine), no checkpoint needed."""

    def __init__(self, dim):
        self.scale = torch.full((dim,), 0.5)
        self.offset = torch.zeros(dim)

    def normalize(self, x):
        return x * self.scale.to(x) + self.offset.to(x)


@click.command()
@click.option("-d", "--device", default="cpu")
@click.option("--N", "n", default=8, show_default=True)
@click.option("--H", "horizon", default=32, show_default=True)
@click.option("--R", "r", default=16, show_default=True, help="executed prefix (n_action_steps)")
@click.option("--warmup", default=200, show_default=True)
@click.option("--reps", default=1000, show_default=True)
@click.option("--trials", default=5, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--out", default=None)
def main(device, n, horizon, r, warmup, reps, trials, seed, out):
    dev = torch.device(device)
    stub = types.SimpleNamespace(action_tokenizer=types.SimpleNamespace(normalizer={"action": _IdNorm(7)}))
    g = torch.Generator().manual_seed(seed)
    cands = [(torch.randn(n, horizon, 7, generator=g) * 0.05).to(dev) for _ in range(16)]
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    rows = {}
    for sig in ("vote", "medoid", "kdpe", "random"):
        gen = torch.Generator().manual_seed(0)
        def call(c, sig=sig):
            return OATPolicy._bon_select(stub, c, r, sig, sel_generator=gen if sig == "random" else None)
        medians = []
        for _ in range(trials):
            for i in range(warmup):
                call(cands[i % 16])
            sync()
            ts = []
            for i in range(reps):
                sync()
                t0 = time.perf_counter()
                call(cands[i % 16])
                sync()
                ts.append((time.perf_counter() - t0) * 1000)
            medians.append(statistics.median(ts))
        rows[sig] = {"median_ms_mean_of_trials": statistics.mean(medians),
                     "std_of_trial_medians": statistics.pstdev(medians), "trial_medians_ms": medians}
        print(f"{sig:8s} {rows[sig]['median_ms_mean_of_trials']:.4f} ms "
              f"(±{rows[sig]['std_of_trial_medians']:.4f}, N={n}, R={r}, device={device})")
    payload = {"N": n, "H": horizon, "R": r, "device": device, "torch": torch.__version__,
               "warmup": warmup, "reps": reps, "trials": trials,
               "note": "includes int() host sync each call; normalizer stand-in for vote/medoid",
               "selectors": rows}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
