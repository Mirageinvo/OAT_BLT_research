"""
Offline diagnostic (NO sim) for the ChunkQ critic + value-BoN selection. Answers "is value-BoN
underperforming because of a code bug or genuine OOD over-estimation?" by mirroring the
inference selection path exactly on STORED features (collect_awr_dataset .npz):

  per state: sample bon_n candidate chunks (same gen as predict_action_bon_free) -> score with
  Q AND with the vote density -> report:
   - within-state Q std (does Q discriminate candidates at all, or argmax = noise?)
   - Spearman corr(Q-rank, vote-rank): >0 value~=vote (should ~match 0.69); <0 value ANTI-vote
     (picks the outliers vote rejects -> explains SR < base) ; ~0 Q uninformative within state
   - Q calibration: mean Q on the success-labeled executed chunk for succ=1 vs succ=0 states
     (cross-state AUC sanity, should separate) — confirms Q itself is fine, isolating the
     within-state failure as the real issue.

Run:
  cd oat && uv run python scripts/diag_chunk_q.py \
      -c my_models/policy_ep-0250_sr-0.596.ckpt -q my_models/chunk_q.ckpt -i my_datasets/awr_bon.npz
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import click
import numpy as np
import torch

from oat.policy.base_policy import BasePolicy
from oat.policy.oatpolicy import OATPolicy
from oat.model.chunk_q import ChunkQ


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-q', '--critic', required=True)
@click.option('-i', '--input', 'inp', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--bon_n', default=8, type=int)
@click.option('--n_states', default=400, type=int)
@click.option('--seed', default=0, type=int)
def main(checkpoint, critic, inp, device, bon_n, n_states, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device(device)
    pol = BasePolicy.from_checkpoint(checkpoint)
    assert isinstance(pol, OATPolicy)
    pol.to(device).eval()
    q = ChunkQ.from_checkpoint(critic).to(device).eval()
    K = pol.max_seq_len
    R = min(pol.n_action_steps, 32)
    norm = pol.action_tokenizer.normalizer['action']

    d = np.load(inp, allow_pickle=True)
    feats_all = d['features'].astype(np.float32)
    succ_all = d['success'].astype(np.float32)
    exec_tok = d['tokens'].astype(np.int64)
    idx = np.random.default_rng(seed).choice(len(feats_all), size=min(n_states, len(feats_all)),
                                              replace=False)

    q_std, sp_corr = [], []
    q_exec_pos, q_exec_neg = [], []
    with torch.inference_mode():
        for j in idx:
            f = torch.from_numpy(feats_all[j])[None].to(device)            # [1, To, d]
            # sample bon_n candidates exactly like predict_action_bon_free
            frep = f.repeat_interleave(bon_n, dim=0)
            bos = torch.full((bon_n, 1), pol.bos_id, dtype=torch.long, device=device)
            tk = pol.model.generate(bos, cond=frep, max_new_tokens=K,
                                    temperature=pol.temperature, top_k=pol.topk)[:, 1:]
            cand = pol.action_tokenizer.detokenize(tk, eval_keep_k=[K] * bon_n)  # [N,H,D]
            a = norm.normalize(cand[:, :R])                                # [N,R,D]
            # Q score per candidate
            feat_rep = f[0][None].expand(bon_n, -1, -1)
            qs = q.score(feat_rep, a).cpu().numpy()                        # [N]
            # vote density rank (mirror _bon_select 'vote')
            flat = a.reshape(bon_n, -1)
            dist = torch.cdist(flat, flat)
            off = dist[dist > 0]
            sigma = (off.median() if off.numel() else dist.new_tensor(1.0)) + 1e-6
            dens = torch.exp(-(dist ** 2) / (2 * sigma ** 2)).sum(1).cpu().numpy()
            q_std.append(qs.std())
            sp_corr.append(spearman(qs, dens))
            # Q on the actually-executed chunk (known success label)
            et = torch.from_numpy(exec_tok[j])[None].to(device)
            ec = pol.action_tokenizer.detokenize(et, eval_keep_k=[K])
            qe = float(q.score(f, norm.normalize(ec[:, :R])).cpu())
            (q_exec_pos if succ_all[j] > 0.5 else q_exec_neg).append(qe)

    print(f"\n=== ChunkQ diagnostic (n_states={len(idx)}, bon_n={bon_n}, R={R}) ===")
    print(f"[within-state] mean Q std across the {bon_n} candidates = {np.mean(q_std):.4f}  "
          f"(near 0 -> Q barely varies within a state -> argmax = noise)")
    sc = np.array(sp_corr)
    print(f"[Q vs vote]    mean spearman(Q-rank, vote-rank) = {sc.mean():+.3f}  "
          f"(>0 value~=vote | <0 value picks vote's REJECTS -> SR<base | ~0 uninformative)")
    print(f"               frac states corr<0 = {np.mean(sc < 0):.2f}")
    mp, mn = np.mean(q_exec_pos), np.mean(q_exec_neg)
    print(f"[Q calib]      Q(executed | succ=1)={mp:.3f}  vs  Q(executed | succ=0)={mn:.3f}  "
          f"gap={mp-mn:+.3f}  (positive gap = Q itself is calibrated cross-state)")
    print("\nREAD: calib gap>0 (Q fine) BUT within-state std~0 or corr<=0 -> not a code bug; "
          "offline Q can't rank WITHIN a state (OOD over-est / no within-state contrast in data). "
          "corr strongly<0 -> Q anti-ranks (prefers OOD outliers) -> explains 0.476<base.")


if __name__ == '__main__':
    main()
