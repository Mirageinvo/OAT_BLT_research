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
import torch
import numpy as np
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
def measure_latency(
    checkpoint: str,
    device: str = 'cuda:0',
    k_tokens: tuple = (),
    warmup: int = 50,
    runs: int = 500,
    batch_size: int = 1,
):
    if not k_tokens:
        k_tokens = (1, 2, 4, 8)

    device = torch.device(device)

    print(f"Loading policy from {checkpoint}...")
    policy, cfg = BasePolicy.from_checkpoint(checkpoint, return_configuration=True)
    policy.to(device)
    policy.eval()

    obs_dict = policy.create_dummy_observation(
        batch_size=batch_size,
        device=device,
    )

    print(f"Observation keys: {list(obs_dict.keys())}")
    print(f"Warmup: {warmup} iters | Timed runs: {runs} iters | Batch size: {batch_size}\n")

    results = {}
    for k in k_tokens:
        # warmup
        with torch.inference_mode():
            for _ in range(warmup):
                policy.predict_action_adaptive(obs_dict, use_k_tokens=k)

        # timed runs
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(runs):
                policy.predict_action_adaptive(obs_dict, use_k_tokens=k)
        torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) / runs * 1000
        results[k] = elapsed_ms
        print(f"OAT({k:2d} tokens): {elapsed_ms:7.2f} ms")

    print(f"\nSpeedup vs OAT(8):")
    ref = results.get(8, results[max(results.keys())])
    for k, ms in results.items():
        print(f"  OAT({k}): {ref / ms:.2f}x faster")


if __name__ == '__main__':
    measure_latency()
