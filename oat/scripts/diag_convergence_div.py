"""
GATE 1 diagnostic: convergence R-signal divergence distribution (NO simulation).

The convergence variable-R signal sets R = leading prefix where the coarse plan
(decode_k = r_coarse_k) agrees with the full plan (decode_K) within r_threshold,
using a per-timestep L2 divergence d_t = ||norm(A_K)_t - norm(A_coarse)_t|| in the
action-normalizer space ([-1,1] per dim, 7 dims).

This script loads real dataset obs, generates K tokens per the policy, decodes the
SAME tokens at coarse-k and K, and reports:
  - the per-timestep divergence distribution d_t (percentiles) -> the natural scale,
    so a sane r_threshold can be chosen (0.5 saturated everything to r_min).
  - for a sweep of candidate thresholds: resulting mean R and the r_exec histogram
    -> decides the GATE question: is R *graded across samples* (heterogeneity exists,
    worth an adaptive controller) or *saturated/bimodal* (no usable signal)?

Usage:
  cd oat && uv run python scripts/diag_convergence_div.py \
      -c my_models/policy_ep-0250_sr-0.596.ckpt --max_samples 4096
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import click
import hydra
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from oat.policy.base_policy import BasePolicy
from oat.policy.oatpolicy import OATPolicy


@click.command()
@click.option('-c', '--checkpoint', required=True, help='policy checkpoint')
@click.option('-d', '--device', default='cuda:0')
@click.option('--batch_size', default=32, type=int)
@click.option('--num_workers', default=4, type=int)
@click.option('--max_samples', default=4096, type=int, help='cap for a quick read')
@click.option('--use_k_tokens', default=8, type=int, help='full budget K')
@click.option('--r_coarse_k', default=4, type=int, help='coarse budget for the signal')
@click.option('--r_min', default=8, type=int)
@click.option('--r_max', default=32, type=int)
@click.option('--greedy', is_flag=True, default=False,
              help='greedy decode (default: sample with policy temperature/topk, '
                   'matching inference)')
def main(checkpoint, device, batch_size, num_workers, max_samples,
         use_k_tokens, r_coarse_k, r_min, r_max, greedy):
    device = torch.device(device)

    policy, cfg = BasePolicy.from_checkpoint(checkpoint, return_configuration=True)
    assert isinstance(policy, OATPolicy), f"Expected OATPolicy, got {type(policy)}"
    policy.to(device).eval()

    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    K = min(use_k_tokens, policy.max_seq_len)
    k_coarse = max(1, min(r_coarse_k, K))
    H = policy.action_tokenizer.latent_horizon
    r_max = min(r_max, H)
    r_min = max(1, min(r_min, r_max))
    norm = policy.action_tokenizer.normalizer['action']
    temperature = policy.temperature
    topk = policy.topk

    d_chunks = []   # per-sample [r_max] divergence
    n_processed = 0
    with torch.inference_mode():
        for batch in tqdm.tqdm(loader):
            obs_dict = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch['obs'].items()
            }
            features = policy.obs_encoder(obs_dict)
            B = features.shape[0]
            bos = torch.full((B, 1), policy.bos_id, dtype=torch.long, device=device)
            if greedy:
                tokens = bos
                for _ in range(K):
                    logits = policy.model(tokens, cond=features)
                    nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    tokens = torch.cat([tokens, nxt], dim=1)
                tokens = tokens[:, 1:]
            else:
                tokens = policy.model.generate(
                    bos, cond=features, max_new_tokens=K,
                    temperature=temperature, top_k=topk,
                )[:, 1:]

            a_full = policy.action_tokenizer.detokenize(tokens, eval_keep_k=[K] * B)
            a_coarse = policy.action_tokenizer.detokenize(tokens, eval_keep_k=[k_coarse] * B)
            d = (norm.normalize(a_full) - norm.normalize(a_coarse)).norm(dim=-1)  # [B,H]
            d_chunks.append(d[:, :r_max].cpu().numpy())

            n_processed += B
            if max_samples is not None and n_processed >= max_samples:
                break

    d = np.concatenate(d_chunks, axis=0)  # [N, r_max]
    N = d.shape[0]
    print(f"\n=== convergence divergence diagnostic (N={N}, K={K}, coarse_k={k_coarse}, "
          f"r in [{r_min},{r_max}]) ===")

    # 1) per-timestep divergence scale
    pcts = [10, 25, 50, 75, 90, 99]
    print("\nper-timestep d_t percentiles (norm-space L2 over 7 dims):")
    print("  t  : " + "  ".join(f"p{p:>2}" for p in pcts))
    for t in range(r_max):
        qs = np.percentile(d[:, t], pcts)
        print(f"  {t:>2} : " + "  ".join(f"{q:5.2f}" for q in qs))

    # 2) candidate-threshold sweep -> resulting R distribution
    def r_exec_for(thr):
        exceed = d > thr                      # [N, r_max]
        any_ex = exceed.any(axis=1)
        first = exceed.argmax(axis=1)         # 0 if none
        r = np.where(any_ex, first, r_max)
        return np.clip(r, r_min, r_max)

    print("\nthreshold sweep -> R distribution (r_min={}, r_max={}):".format(r_min, r_max))
    print(f"  {'thr':>5} | {'meanR':>6} {'medR':>5} {'%@min':>6} {'%@max':>6} {'stdR':>5} | hist")
    for thr in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]:
        r = r_exec_for(thr)
        at_min = 100.0 * (r == r_min).mean()
        at_max = 100.0 * (r == r_max).mean()
        # coarse histogram in 4 bins across [r_min, r_max]
        edges = np.linspace(r_min, r_max, 5)
        hist = np.histogram(r, bins=edges)[0]
        print(f"  {thr:5.1f} | {r.mean():6.2f} {np.median(r):5.0f} "
              f"{at_min:6.1f} {at_max:6.1f} {r.std():5.2f} | {hist.tolist()}")

    print("\nGATE read:")
    print("  - %@min and %@max both large at the 'mean R~16' threshold => BIMODAL")
    print("    (some chunks short, some long) -> heterogeneity EXISTS, adaptive R has")
    print("    something to exploit. Run the matched-mean sim: convergence vs random.")
    print("  - all mass at one end across every threshold => SATURATED -> no signal")
    print("    (K-gate deja vu); the convergence cue is moot regardless of sim.")


if __name__ == '__main__':
    main()
