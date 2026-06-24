"""
Train a ChunkQ critic (our Q-chunking QC analog) on a collect_awr_dataset rollout dump.

Each executed chunk is labeled by its EPISODE success (Monte-Carlo return, gamma=1, sparse
terminal reward) -> Q(features, chunk) ~ P(success | state, chunk) under the data policy.
Tokens are detokenized through the frozen OAT tokenizer to the continuous chunk, normalized to
normalizer space (matching OATPolicy._bon_select / the inference geometry), and the first
`horizon` (= n_action_steps) steps are scored. Train/val split is by EPISODE (no leakage).

At inference, attach the saved critic (eval_policy_sim --chunk_q ... --bon_signal value) and the
policy executes argmax_chunk Q over the N best-of-N candidates instead of the consensus vote.

Run:
  cd oat && uv run python scripts/train_chunk_q.py \
      -i my_datasets/awr_bon.npz -c my_models/policy_ep-0250_sr-0.596.ckpt \
      -o my_models/chunk_q.ckpt --epochs 30
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import click
import numpy as np
import torch
import torch.nn as nn

from oat.policy.base_policy import BasePolicy
from oat.policy.oatpolicy import OATPolicy
from oat.model.chunk_q import ChunkQ


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC of scores s for binary labels y, via the rank statistic (no sklearn)."""
    pos, neg = y > 0.5, y <= 0.5
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    ranks = s.argsort().argsort().astype(np.float64) + 1.0   # average-tie ignored (cheap)
    return (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


@torch.inference_mode()
def detok_chunks(policy, tokens, horizon, device, bs=512):
    """tokens [N,K] long -> normalized chunk [N, horizon, D] (normalizer space)."""
    K = policy.max_seq_len
    norm = policy.action_tokenizer.normalizer['action']
    out = []
    for i in range(0, tokens.shape[0], bs):
        tk = torch.from_numpy(tokens[i:i + bs]).long().to(device)
        a = policy.action_tokenizer.detokenize(tk, eval_keep_k=[K] * tk.shape[0])  # [b,H,D]
        out.append(norm.normalize(a[:, :horizon]).cpu())
    return torch.cat(out, 0)


@click.command()
@click.option('-i', '--input', 'inp', required=True, help='collect_awr_dataset .npz')
@click.option('-c', '--checkpoint', required=True, help='OAT policy ckpt (detokenize+normalizer)')
@click.option('-o', '--output', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--horizon', default=None, type=int, help='scored chunk steps R (default = policy.n_action_steps)')
@click.option('--epochs', default=30, type=int)
@click.option('--batch_size', default=512, type=int)
@click.option('--lr', default=1e-3, type=float)
@click.option('--weight_decay', default=1e-4, type=float)
@click.option('--val_frac', default=0.15, type=float)
@click.option('--seed', default=0, type=int)
def main(inp, checkpoint, output, device, horizon, epochs, batch_size, lr,
         weight_decay, val_frac, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device(device)
    policy = BasePolicy.from_checkpoint(checkpoint)
    assert isinstance(policy, OATPolicy)
    policy.to(device).eval()
    if horizon is None:
        horizon = policy.n_action_steps

    d = np.load(inp, allow_pickle=True)
    feats = d['features'].astype(np.float32)            # [N, To, dd]
    succ = d['success'].astype(np.float32)              # [N]
    ep = d['ep_id']                                     # [N]
    N, To, dd = feats.shape
    chunks = detok_chunks(policy, d['tokens'], horizon, device)   # [N, horizon, D] (cpu)
    D = chunks.shape[-1]
    print(f"N={N}  features [{To},{dd}]  chunk [{horizon},{D}]  base SR={succ.mean():.3f}")

    # split by EPISODE (no chunk leakage across train/val)
    uep = np.unique(ep)
    rng = np.random.default_rng(seed)
    rng.shuffle(uep)
    n_val = max(1, int(len(uep) * val_frac))
    val_ep = set(uep[:n_val].tolist())
    is_val = np.array([e in val_ep for e in ep])
    tr, va = ~is_val, is_val
    print(f"train chunks={int(tr.sum())} ({succ[tr].mean():.3f} SR)  "
          f"val chunks={int(va.sum())} ({succ[va].mean():.3f} SR)  over {len(uep)} episodes")

    feats_t = torch.from_numpy(feats)
    succ_t = torch.from_numpy(succ)
    tr_idx = torch.from_numpy(np.where(tr)[0])
    va_idx = torch.from_numpy(np.where(va)[0])

    model = ChunkQ(in_dim=dd, n_obs_steps=To, horizon=horizon, action_dim=D).to(device)
    fm = feats_t[tr_idx].reshape(len(tr_idx), -1).mean(0)
    fs = feats_t[tr_idx].reshape(len(tr_idx), -1).std(0)
    model.set_feature_stats(fm, fs)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # class-balance the sparse-ish success label
    pos_w = torch.tensor([(succ[tr] <= 0.5).sum() / max(1, (succ[tr] > 0.5).sum())], device=device)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    print(f"pos_weight={pos_w.item():.2f}")

    def run_eval():
        model.eval()
        with torch.inference_mode():
            logits = []
            for i in range(0, len(va_idx), 4096):
                ix = va_idx[i:i + 4096]
                logits.append(model(feats_t[ix].to(device), chunks[ix].to(device)).cpu())
            lg = torch.cat(logits).numpy()
        y = succ[va_idx.numpy()]
        bce = nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(lg), torch.from_numpy(y)).item()
        return bce, auc(y, lg), float((1 / (1 + np.exp(-lg))).std())

    best_auc, best_state = -1.0, None
    for ep_i in range(epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        tot = 0.0
        for i in range(0, len(perm), batch_size):
            ix = perm[i:i + batch_size]
            logit = model(feats_t[ix].to(device), chunks[ix].to(device))
            loss = lossfn(logit, succ_t[ix].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(ix)
        bce, a, sstd = run_eval()
        tag = ''
        if a > best_auc:
            best_auc, best_state = a, {k: v.detach().cpu().clone()
                                       for k, v in model.state_dict().items()}
            tag = '  *best'
        print(f"ep {ep_i:3d}  train_loss {tot/len(perm):.4f}  val_bce {bce:.4f}  "
              f"val_auc {a:.4f}  val_pred_std {sstd:.3f}{tag}")

    model.load_state_dict(best_state)
    cfg = dict(in_dim=dd, n_obs_steps=To, horizon=horizon, action_dim=D,
               hidden_dims=(256, 256), dropout=0.1)
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'config': cfg, 'model_state': model.state_dict()}, output)
    print(f"\nsaved best (val_auc={best_auc:.4f}) -> {output}")
    print("NB val_auc>0.5 = Q separates success from features+chunk; val_pred_std>~0.05 = Q "
          "varies enough across inputs to rank candidates. Decisive test = value-BoN in sim.")


if __name__ == '__main__':
    main()
