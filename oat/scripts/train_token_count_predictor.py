"""
Train the token-count predictor for adaptive-budget OATPolicy.

Consumes the .npz produced by `scripts/collect_min_k_dataset.py`:
  features [N, To, d], errors [N, max_k], optional task_uids.

Label: min_k(eps) = first k (1-based) with errors[:, k-1] < eps, else max_k
(a "fail" -- even the full budget did not land within eps of the demo; we
conservatively label it as the full budget so the policy keeps all tokens).
Class index = min_k - 1.

The key asymmetry to keep in mind when reading the metrics: UNDER-predicting k
(fewer tokens than needed) degrades the action and likely task success, while
OVER-predicting only costs latency. So `under_rate` matters more than raw
accuracy. Plain CE is the v1 baseline; see `--underpredict_weight` for a simple
safety bias.

Usage:
  python scripts/train_token_count_predictor.py \
      --dataset my_datasets/libero10_min_k_features.npz \
      --output my_models/token_count_predictor.ckpt \
      --epsilon 0.10
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import pathlib
import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from oat.model.token_count_predictor import TokenCountPredictor


def build_labels(errors: np.ndarray, eps: float):
    """errors [N, max_k] -> (labels [N] in [0, max_k-1], is_fail [N] bool)."""
    below = errors < eps                       # [N, max_k]
    has_below = below.any(axis=1)              # [N]
    max_k = errors.shape[1]
    # argmax on bool returns first True; for all-False rows -> 0, overwrite with max_k-1
    first_below = below.argmax(axis=1)
    labels = np.where(has_below, first_below, max_k - 1).astype(np.int64)
    return labels, ~has_below


@torch.no_grad()
def evaluate(model, loader, device, max_k):
    model.eval()
    preds, trues = [], []
    for feats, labels in loader:
        logits = model(feats.to(device))
        preds.append(logits.argmax(-1).cpu())
        trues.append(labels)
    pred = torch.cat(preds).numpy() + 1        # k in [1, max_k]
    true = torch.cat(trues).numpy() + 1
    n = len(true)

    exact = (pred == true).mean()
    under = (pred < true).mean()               # too few tokens -> risky
    over = (pred > true).mean()                # too many -> just slower
    safe = (pred >= true).mean()               # enough tokens
    extra = (pred - true).mean()               # avg token delta vs oracle
    return {
        "exact_acc": float(exact),
        "under_rate": float(under),
        "over_rate": float(over),
        "safe_rate": float(safe),
        "mean_pred_k": float(pred.mean()),
        "mean_true_k": float(true.mean()),
        "mean_extra_tokens": float(extra),
        "pred_hist": np.bincount(pred, minlength=max_k + 1)[1:].tolist(),
        "true_hist": np.bincount(true, minlength=max_k + 1)[1:].tolist(),
    }


@click.command()
@click.option('-i', '--dataset', required=True, help='.npz from collect_min_k_dataset.py')
@click.option('-o', '--output', required=True, help='output .ckpt path')
@click.option('--epsilon', default=0.10, type=float, help='error threshold for min_k labels')
@click.option('--hidden_dims', default='256,256', help='comma-separated MLP hidden dims')
@click.option('--dropout', default=0.1, type=float)
@click.option('--epochs', default=100, type=int)
@click.option('--batch_size', default=512, type=int)
@click.option('--lr', default=1e-3, type=float)
@click.option('--weight_decay', default=1e-4, type=float)
@click.option('--val_ratio', default=0.1, type=float)
@click.option('--underpredict_weight', default=1.0, type=float,
              help='>1 up-weights CE loss on under-predicted samples (safety bias)')
@click.option('--device', default='cuda:0')
@click.option('--seed', default=42, type=int)
def main(dataset, output, epsilon, hidden_dims, dropout, epochs, batch_size,
         lr, weight_decay, val_ratio, underpredict_weight, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)

    data = np.load(dataset, allow_pickle=True)
    features = data['features'].astype(np.float32)     # [N, To, d]
    errors = data['errors'].astype(np.float32)         # [N, max_k]
    N, n_obs_steps, in_dim = features.shape
    max_k = errors.shape[1]
    print(f"Loaded {N} samples | features [N,{n_obs_steps},{in_dim}] | max_k={max_k}")

    labels, is_fail = build_labels(errors, epsilon)
    print(f"eps={epsilon} | fail (label=max_k): {is_fail.mean():.3f} | "
          f"label hist (k=1..{max_k}): {np.bincount(labels + 1, minlength=max_k + 1)[1:].tolist()}")

    # train/val split
    perm = np.random.permutation(N)
    n_val = int(N * val_ratio)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    feats_t = torch.from_numpy(features)
    labels_t = torch.from_numpy(labels)

    # standardization stats from TRAIN split only (flattened To*d)
    train_flat = feats_t[train_idx].reshape(len(train_idx), -1)
    feat_mean = train_flat.mean(0)
    feat_std = train_flat.std(0)

    model = TokenCountPredictor(
        in_dim=in_dim, n_obs_steps=n_obs_steps, num_classes=max_k,
        hidden_dims=tuple(int(x) for x in hidden_dims.split(',')), dropout=dropout,
    ).to(device)
    model.set_feature_stats(feat_mean.to(device), feat_std.to(device))

    train_loader = DataLoader(
        TensorDataset(feats_t[train_idx], labels_t[train_idx]),
        batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TensorDataset(feats_t[val_idx], labels_t[val_idx]),
        batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_safe, best_state, best_metrics = -1.0, None, None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for feats, lbl in train_loader:
            feats, lbl = feats.to(device), lbl.to(device)
            logits = model(feats)
            loss = F.cross_entropy(logits, lbl, reduction='none')
            if underpredict_weight != 1.0:
                # up-weight samples the model currently under-predicts
                under = (logits.argmax(-1) < lbl).float()
                loss = loss * (1.0 + (underpredict_weight - 1.0) * under)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.shape[0]
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs - 1:
            m = evaluate(model, val_loader, device, max_k)
            print(f"ep {epoch:3d} | loss {total_loss / len(train_idx):.4f} | "
                  f"acc {m['exact_acc']:.3f} | under {m['under_rate']:.3f} | "
                  f"safe {m['safe_rate']:.3f} | pred_k {m['mean_pred_k']:.2f} "
                  f"(oracle {m['mean_true_k']:.2f}, full {max_k})")
            # select on safe_rate, tie-break by fewer mean tokens
            score = m['safe_rate'] - 1e-3 * m['mean_pred_k']
            if score > best_safe:
                best_safe = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = m

    print("\nBest val metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v}")
    print(f"\nLatency proxy: predictor uses {best_metrics['mean_pred_k']:.2f} tokens/step "
          f"vs full budget {max_k} "
          f"(~{(1 - best_metrics['mean_pred_k'] / max_k) * 100:.0f}% fewer AR steps), "
          f"under-predicting on {best_metrics['under_rate'] * 100:.1f}% of samples.")

    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': best_state,
        'config': {
            'in_dim': in_dim, 'n_obs_steps': n_obs_steps, 'num_classes': max_k,
            'hidden_dims': tuple(int(x) for x in hidden_dims.split(',')), 'dropout': dropout,
        },
        'epsilon': epsilon,
        'val_metrics': best_metrics,
    }, output)
    print(f"\nSaved predictor to {output}")


if __name__ == '__main__':
    main()
