"""
Measure inference latency for OAT policy with different numbers of tokens.

Usage:
    uv run scripts/measure_latency.py --checkpoint path/to/policy.ckpt
    uv run scripts/measure_latency.py --checkpoint path/to/policy.ckpt --k_tokens 1 2 4 8
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import time
import click
import hydra
import torch
import numpy as np
from torch.utils.data import DataLoader
from oat.policy.base_policy import BasePolicy
from typing import List, Optional


@click.command()
@click.option('-c', '--checkpoint', required=True, help="path to .ckpt file")
@click.option('-d', '--device', default='cuda:0', help="device to run on")
@click.option('--k_tokens', default=None, multiple=True, type=int,
              help="number of tokens to benchmark (default: 1 2 4 8)")
@click.option('--warmup', default=50, help="number of warmup iterations")
@click.option('--runs', default=500, help="number of timed iterations per k")
@click.option('--batch_size', default=1, help="batch size for inference")
@click.option('--obs_from_checkpoint_dataset', is_flag=True, default=False,
              help="load real obs from the checkpoint's validation dataset instead of random noise")
@click.option('--n_batches', default=10, type=int,
              help="number of distinct obs batches to cycle through during timing")
def measure_latency(
    checkpoint: str,
    device: str = 'cuda:0',
    k_tokens: tuple = (),
    warmup: int = 50,
    runs: int = 500,
    batch_size: int = 1,
    obs_from_checkpoint_dataset: bool = False,
    n_batches: int = 10,
):
    if not k_tokens:
        k_tokens = (1, 2, 4, 8)

    device = torch.device(device)

    print(f"Loading policy from {checkpoint}...")
    policy, cfg = BasePolicy.from_checkpoint(checkpoint, return_configuration=True)
    policy.to(device)
    policy.eval()

    if obs_from_checkpoint_dataset:
        print("Instantiating validation dataset from checkpoint config...")
        dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
        val_dataset = dataset.get_validation_dataset()
        loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        ports = policy.get_observation_ports()
        obs_batches = []
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            obs_batches.append({k: batch['obs'][k].to(device) for k in ports})
        print(f"Loaded {len(obs_batches)} obs batches from val dataset (of {len(val_dataset)} samples)")
    else:
        obs_batches = [policy.create_dummy_observation(
            batch_size=batch_size,
            device=device,
        )]

    print(f"Observation keys: {list(obs_batches[0].keys())}")
    print(f"Source: {'val dataset' if obs_from_checkpoint_dataset else 'random noise (dummy)'}")
    print(f"Warmup: {warmup} iters | Timed runs: {runs} iters | Batch size: {batch_size} | Batches: {len(obs_batches)}\n")

    n_obs = len(obs_batches)
    results = {}
    entropies_per_cap = {}
    for k in k_tokens:
        # warmup
        with torch.inference_mode():
            for i in range(warmup):
                policy.predict_action_adaptive(obs_batches[i % n_obs], use_k_tokens=k)

        # timed runs
        token_counts = []
        entropies_by_step = [[] for _ in range(k)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            for i in range(runs):
                out = policy.predict_action_adaptive(obs_batches[i % n_obs], use_k_tokens=k)
                token_counts.append(out['n_tokens'])
                for step_idx, e in enumerate(out['entropies']):
                    entropies_by_step[step_idx].append(e)
        torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) / runs * 1000
        tokens_arr = np.array(token_counts)
        avg_tokens = tokens_arr.mean()
        results[k] = (elapsed_ms, avg_tokens, tokens_arr)
        entropies_per_cap[k] = entropies_by_step
        print(f"OAT(cap={k:2d}): {elapsed_ms:7.2f} ms | tokens: avg={avg_tokens:.2f} "
              f"min={tokens_arr.min()} max={tokens_arr.max()} median={int(np.median(tokens_arr))}")

    # entropy percentiles per step (use the largest cap run — it has all steps populated)
    k_ref = max(entropies_per_cap.keys())
    print(f"\nEntropy percentiles per step (from cap={k_ref} run, n={runs}):")
    print(f"  {'step':>4} {'n':>5} {'p10':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p90':>7} {'min':>7} {'max':>7}")
    for step_idx, es in enumerate(entropies_per_cap[k_ref]):
        if not es:
            continue
        arr = np.array(es)
        p10, p25, p50, p75, p90 = np.percentile(arr, [10, 25, 50, 75, 90])
        print(f"  {step_idx:>4d} {len(arr):>5d} {p10:>7.3f} {p25:>7.3f} {p50:>7.3f} "
              f"{p75:>7.3f} {p90:>7.3f} {arr.min():>7.3f} {arr.max():>7.3f}")

    print(f"\nSpeedup vs OAT(cap=8):")
    ref_ms, _, _ = results.get(8, results[max(results.keys())])
    for k, (ms, avg, _) in results.items():
        print(f"  OAT(cap={k}, avg={avg:.2f}): {ref_ms / ms:.2f}x faster")


if __name__ == '__main__':
    measure_latency()
