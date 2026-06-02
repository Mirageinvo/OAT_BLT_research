"""
Step 0 for H-OAT: is the *marginal value of tokens* unevenly distributed WITHIN an
action chunk? (premise check for hierarchical / execution-aware OAT)

For each demo action chunk [H, D], autoencode through the frozen OAT tokenizer at budget
k and measure per-timestep error e_t(k) = RMS_dim(norm(recon_k[t]) - norm(gt[t])).

Key point (raw error is NOT enough): H-OAT redistributes token BUDGET, so what matters is
where extra tokens actually *help*, not where error is merely high. A timestep can be hard
at every k (high floor) -> more tokens won't help. So we also measure:
  excess_t(k) = max(e_t(k) - e_t(k_max), 0)   # headroom at budget k (under-served-ness)
  gain_t(a->b) = max(e_t(a) - e_t(b), 0)       # marginal improvement from a->b tokens
Concentration of excess/gain (not raw e) is the direct argument for H-OAT.

CAUTION — averaging over chunks SMEARS within-chunk peaks (contact is at different absolute
times). So we report PER-CHUNK unevenness (averaged over chunks), not the mean profile:
  CV = std_t/mean_t ; peak2mean = max_t/mean_t ; topN_conc = sum(N largest)/sum_t
Uniform refs: CV=0, peak2mean=1, topN_conc = N/L (L = H full, or R window).
Also reported restricted to the executed window [:exec_window] (execution-aware view),
where the right uniform baseline is N/R (e.g. top4 over R=8 is already 0.5).

Usage:
  python scripts/per_timestep_recon_error.py -c my_models/policy_ep-0250_sr-0.596.ckpt
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
@click.option('-c', '--checkpoint', required=True, help='policy checkpoint (has frozen tokenizer)')
@click.option('-d', '--device', default='cuda:0')
@click.option('--batch_size', default=64, type=int)
@click.option('--num_workers', default=4, type=int)
@click.option('--max_samples', default=20000, type=int, help='cap for speed (None = full)')
@click.option('--budgets', default='1,2,4,8', help='token budgets to probe (sorted; last = reference)')
@click.option('--exec_window', default=8, type=int, help='executed window R (execution-aware view)')
def main(checkpoint, device, batch_size, num_workers, max_samples, budgets, exec_window):
    device = torch.device(device)
    policy, cfg = BasePolicy.from_checkpoint(checkpoint, return_configuration=True)
    assert isinstance(policy, OATPolicy)
    policy.to(device).eval()

    tok = policy.action_tokenizer
    normalizer = tok.normalizer['action']
    budgets = sorted(int(b) for b in budgets.split(','))
    k_max = budgets[-1]

    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    print(f"Dataset size: {len(dataset)} | budgets={budgets} | exec_window={exec_window}")

    eps = 1e-8
    H = None
    R = None
    sums = {}          # name -> {cv,p2m,top2,top4, cv_w,p2m_w,top2_w,top4_w}
    raw_profile = {}   # k -> sum of e_t over chunks (reference only)
    n_chunks = 0

    def chunk_stats(ee):  # ee: [B, L] -> (cv, p2m, top2, top4) averaged over batch
        mean_t = ee.mean(1)
        cv = (ee.std(1) / (mean_t + eps)).mean().item()
        p2m = (ee.max(1).values / (mean_t + eps)).mean().item()
        tot = ee.sum(1) + eps
        top2 = (ee.topk(min(2, ee.shape[1]), 1).values.sum(1) / tot).mean().item()
        top4 = (ee.topk(min(4, ee.shape[1]), 1).values.sum(1) / tot).mean().item()
        return cv, p2m, top2, top4

    def add(name, ee, B):
        if name not in sums:
            sums[name] = dict(cv=0, p2m=0, top2=0, top4=0, cv_w=0, p2m_w=0, top2_w=0, top4_w=0)
        c = chunk_stats(ee)
        cw = chunk_stats(ee[:, :R])
        s = sums[name]
        for key, v in zip(('cv', 'p2m', 'top2', 'top4'), c):
            s[key] += v * B
        for key, v in zip(('cv_w', 'p2m_w', 'top2_w', 'top4_w'), cw):
            s[key] += v * B

    with torch.inference_mode():
        for batch in tqdm.tqdm(loader):
            gt = batch['action'].to(device, non_blocking=True)   # [B, H, D]
            B = gt.shape[0]
            if H is None:
                H = gt.shape[1]
                R = min(exec_window, H)
            gt_n = normalizer.normalize(gt)

            e = {}
            for k in budgets:
                recon_n = normalizer.normalize(tok.autoencode(gt, eval_keep_k=[k] * B))
                e[k] = (recon_n - gt_n).pow(2).mean(-1).sqrt()   # [B, H]
                raw_profile[k] = raw_profile.get(k, np.zeros(H)) + e[k].sum(0).cpu().numpy()

            for k in budgets:
                add(f'e_k{k}', e[k], B)
            for k in budgets[:-1]:
                add(f'excess_k{k}', (e[k] - e[k_max]).clamp(min=0), B)
            for a, b in zip(budgets[:-1], budgets[1:]):
                add(f'gain_{a}to{b}', (e[a] - e[b]).clamp(min=0), B)

            n_chunks += B
            if max_samples is not None and n_chunks >= max_samples:
                break

    n = n_chunks
    print(f"\nProcessed {n} chunks | H={H} | exec_window R={R}")
    print(f"Uniform refs — full(H={H}): top2={2/H:.3f} top4={4/H:.3f} | "
          f"window(R={R}): top2={2/R:.3f} top4={4/R:.3f} | CV=0 peak2mean=1\n")

    def show(name):
        s = sums[name]
        print(f"  {name:14s} | full:   CV {s['cv']/n:.2f}  p2m {s['p2m']/n:.2f}  "
              f"top2 {s['top2']/n:.3f}  top4 {s['top4']/n:.3f}")
        print(f"  {'':14s} | window: CV {s['cv_w']/n:.2f}  p2m {s['p2m_w']/n:.2f}  "
              f"top2 {s['top2_w']/n:.3f}  top4 {s['top4_w']/n:.3f}")

    print("--- RAW error e_t(k) (high error != needs more tokens) ---")
    for k in budgets:
        show(f'e_k{k}')
    print("\n--- EXCESS_t(k) = e_t(k) - e_t(kmax)  [headroom: where budget k is under-served] ---")
    for k in budgets[:-1]:
        show(f'excess_k{k}')
    print("\n--- GAIN_t(a->b)  [marginal improvement from doubling tokens] ---")
    for a, b in zip(budgets[:-1], budgets[1:]):
        show(f'gain_{a}to{b}')

    print("\n--- mean per-step err profile (SMEARED, reference only) ---")
    for k in budgets:
        prof = raw_profile[k] / n
        print(f"  k={k}: overall {prof.mean():.4f} | window {prof[:R].mean():.4f} | tail {prof[R:].mean():.4f}")


if __name__ == '__main__':
    main()
