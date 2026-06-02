"""
Step 0 for H-OAT: is the *marginal value of tokens* unevenly distributed WITHIN an
action chunk, and does it coincide with contact/grasp/jerk? (premise check)

For each demo chunk [H, D], autoencode through the frozen OAT tokenizer at budget k and
measure per-timestep error e_t(k) = RMS_dim(norm(recon_k[t]) - norm(gt[t])). Raw error is
not enough (a step can be hard at every k); what matters for redistributing BUDGET is the
marginal value of tokens:
  gain_t(a->b) = max(e_t(a) - e_t(b), 0)
  excess_t(k)  = max(e_t(k) - e_t(kmax), 0)

Reported (per-chunk, then aggregated; mean profile would SMEAR within-chunk peaks):
  CV, peak2mean, top2_conc, top4_conc   (uniform refs: CV0, p2m1, topN = N/L)
for full chunk and for executed windows R (execution-aware). Plus:
  - QUANTILES of gain_4to8 concentration (median/p75/p90), not just mean.
  - Per-chunk correlation of gain_4to8 with gripper-change / action-delta / jerk, and the
    ratio of those features at the top-gain step vs chunk mean (semantic validation).

Usage:
  python scripts/per_timestep_recon_error.py -c my_models/policy_ep-0250_sr-0.596.ckpt
"""

if __name__ == "__main__":
    import sys, os, pathlib
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

EPS = 1e-8


def chunk_stats(ee):  # ee: [B, L] -> per-chunk (cv, p2m, top2, top4) tensors [B]
    mean_t = ee.mean(1)
    cv = ee.std(1) / (mean_t + EPS)
    p2m = ee.max(1).values / (mean_t + EPS)
    tot = ee.sum(1) + EPS
    top2 = ee.topk(min(2, ee.shape[1]), 1).values.sum(1) / tot
    top4 = ee.topk(min(4, ee.shape[1]), 1).values.sum(1) / tot
    return cv, p2m, top2, top4


def per_chunk_corr(x, y):  # x,y: [B, H] -> per-chunk Pearson r [B]
    xc = x - x.mean(1, keepdim=True)
    yc = y - y.mean(1, keepdim=True)
    num = (xc * yc).sum(1)
    den = xc.norm(dim=1) * yc.norm(dim=1) + EPS
    return num / den


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--batch_size', default=64, type=int)
@click.option('--num_workers', default=4, type=int)
@click.option('--max_samples', default=20000, type=int)
@click.option('--budgets', default='1,2,4,8')
@click.option('--exec_windows', default='4,8,16', help='executed windows R to report')
@click.option('--gripper_dim', default=-1, type=int, help='action dim index of the gripper')
def main(checkpoint, device, batch_size, num_workers, max_samples, budgets, exec_windows, gripper_dim):
    device = torch.device(device)
    policy, cfg = BasePolicy.from_checkpoint(checkpoint, return_configuration=True)
    assert isinstance(policy, OATPolicy)
    policy.to(device).eval()
    tok = policy.action_tokenizer
    normalizer = tok.normalizer['action']
    budgets = sorted(int(b) for b in budgets.split(','))
    k_max = budgets[-1]
    windows = sorted(int(r) for r in exec_windows.split(','))

    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    print(f"Dataset {len(dataset)} | budgets={budgets} | windows={windows}")

    H = None
    sums = {}        # (name, scope) -> [cv,p2m,top2,top4] running sums
    n = 0
    quant = {}       # name -> list of per-chunk top4 (full)  for quantiles
    corr = {f: 0.0 for f in ('gripper', 'delta', 'jerk')}
    topratio = {f: 0.0 for f in ('gripper', 'delta', 'jerk')}

    def acc(name, ee, scope):
        c = [v.sum().item() for v in chunk_stats(ee)]
        key = (name, scope)
        if key not in sums:
            sums[key] = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            sums[key][i] += c[i]

    with torch.inference_mode():
        for batch in tqdm.tqdm(loader):
            gt = batch['action'].to(device, non_blocking=True)   # [B,H,D]
            B, Hb, D = gt.shape
            if H is None:
                H = Hb
            gt_n = normalizer.normalize(gt)

            e = {k: (normalizer.normalize(tok.autoencode(gt, eval_keep_k=[k] * B)) - gt_n)
                 .pow(2).mean(-1).sqrt() for k in budgets}   # [B,H]

            quantities = {}
            for k in budgets:
                quantities[f'e_k{k}'] = e[k]
            for k in budgets[:-1]:
                quantities[f'excess_k{k}'] = (e[k] - e[k_max]).clamp(min=0)
            for a, b in zip(budgets[:-1], budgets[1:]):
                quantities[f'gain_{a}to{b}'] = (e[a] - e[b]).clamp(min=0)

            for name, ee in quantities.items():
                acc(name, ee, 'full')
                for R in windows:
                    acc(name, ee, f'R{R}')

            # quantiles for the decisive quantity
            g48 = quantities[f'gain_{budgets[-2]}to{k_max}']
            quant.setdefault('gain_top4_full', []).append(chunk_stats(g48)[3].cpu().numpy())
            for R in windows:
                quant.setdefault(f'gain_top2_R{R}', []).append(chunk_stats(g48[:, :R])[2].cpu().numpy())

            # semantic features (normalized space)
            gi = gripper_dim
            delta = torch.zeros_like(e[k_max]); delta[:, 1:] = (gt_n[:, 1:] - gt_n[:, :-1]).norm(dim=-1); delta[:, 0] = delta[:, 1]
            jerk = torch.zeros_like(e[k_max]); jerk[:, 2:] = (gt_n[:, 2:] - 2 * gt_n[:, 1:-1] + gt_n[:, :-2]).norm(dim=-1); jerk[:, :2] = jerk[:, 2:3]
            grip = torch.zeros_like(e[k_max]); grip[:, 1:] = (gt_n[:, 1:, gi] - gt_n[:, :-1, gi]).abs(); grip[:, 0] = grip[:, 1]
            feats = {'gripper': grip, 'delta': delta, 'jerk': jerk}
            top_idx = g48.argmax(1)
            ar = torch.arange(B, device=device)
            for f, fv in feats.items():
                corr[f] += per_chunk_corr(g48, fv).nan_to_num(0).sum().item()
                topratio[f] += (fv[ar, top_idx] / (fv.mean(1) + EPS)).sum().item()

            n += B
            if max_samples is not None and n >= max_samples:
                break

    print(f"\nProcessed {n} chunks | H={H}")
    refs = {'full': H}
    for R in windows:
        refs[f'R{R}'] = R
    print("Uniform refs (topN = N/L):")
    for scope, L in refs.items():
        print(f"  {scope:5s} (L={L}): top2={2/L:.3f} top4={4/L:.3f}")

    def block(title, names):
        print(f"\n--- {title} ---")
        print(f"  {'quantity':12s} {'scope':5s} | CV    p2m   top2   top4")
        for name in names:
            for scope in ['full'] + [f'R{R}' for R in windows]:
                if (name, scope) in sums:
                    s = sums[(name, scope)]
                    print(f"  {name:12s} {scope:5s} | {s[0]/n:.2f}  {s[1]/n:.2f}  {s[2]/n:.3f}  {s[3]/n:.3f}")

    block("RAW e_t(k)", [f'e_k{k}' for k in budgets])
    block("EXCESS_t(k) = e(k)-e(kmax)", [f'excess_k{k}' for k in budgets[:-1]])
    block("GAIN_t(a->b)", [f'gain_{a}to{b}' for a, b in zip(budgets[:-1], budgets[1:])])

    print(f"\n--- QUANTILES of gain_{budgets[-2]}to{k_max} concentration ---")
    for key, arrs in quant.items():
        v = np.concatenate(arrs)
        print(f"  {key:16s}: median {np.median(v):.3f}  p75 {np.percentile(v,75):.3f}  "
              f"p90 {np.percentile(v,90):.3f}  mean {v.mean():.3f}")

    print(f"\n--- SEMANTICS: gain_{budgets[-2]}to{k_max} vs action features (within-chunk) ---")
    print(f"  feature   | per-chunk Pearson r | feature@top-gain-step / chunk-mean")
    for f in ('gripper', 'delta', 'jerk'):
        print(f"  {f:9s} |  {corr[f]/n:+.3f}             |  {topratio[f]/n:.2f}x")

    print("\nRead: H-OAT premise strong if GAIN top2/top4 >> refs WITH high median (not just")
    print("mean), and gain correlates with gripper/jerk (semantic) — especially within window.")


if __name__ == '__main__':
    main()
