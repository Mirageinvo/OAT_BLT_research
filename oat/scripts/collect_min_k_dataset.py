"""
Collect per-sample error labels for training a token-count predictor.

For each sample in the policy's training dataset:
  - encode obs -> features
  - greedily generate all max_seq_len action tokens
  - for k=1..max_seq_len: detokenize the first k tokens, compute
        err(k) = sqrt( mean( (normalize(a_pred(k)) - normalize(a_gt))^2 ) )
    i.e. RMS per-dim error in the action-normalizer space ([-1, 1]
    per dim), so eps is interpretable as the average fraction of each
    dim's data range that predictions deviate from ground truth.
  - persist (features, errors, task_uid).

Usage:
  python scripts/collect_min_k_dataset.py \
      --checkpoint my_models/policy_ep-0250_sr-0.596.ckpt \
      --output data/libero/libero10_min_k_features.npz
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
import hydra
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from oat.policy.base_policy import BasePolicy
from oat.policy.oatpolicy import OATPolicy


@click.command()
@click.option('-c', '--checkpoint', required=True, help='policy checkpoint')
@click.option('-o', '--output', required=True, help='output .npz path')
@click.option('-d', '--device', default='cuda:0')
@click.option('--batch_size', default=16, type=int)
@click.option('--num_workers', default=4, type=int)
@click.option('--max_samples', default=None, type=int, help='cap for quick debug')
@click.option('--epsilon', default=0.10, type=float, help='relative error threshold for stats')
def main(checkpoint, output, device, batch_size, num_workers, max_samples, epsilon):
    device = torch.device(device)

    policy, cfg = BasePolicy.from_checkpoint(checkpoint, return_configuration=True)
    assert isinstance(policy, OATPolicy), f"Expected OATPolicy, got {type(policy)}"
    policy.to(device).eval()

    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    max_k = policy.max_seq_len
    n_action_steps = policy.n_action_steps
    action_normalizer = policy.action_tokenizer.normalizer['action']

    feature_chunks = []
    error_chunks = []
    task_uid_chunks = []

    n_processed = 0
    with torch.inference_mode():
        for batch in tqdm.tqdm(loader):
            obs_dict = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch['obs'].items()
            }
            gt_action = batch['action'].to(device, non_blocking=True)  # [B, Ta, Da]
            B = gt_action.shape[0]

            features = policy.obs_encoder(obs_dict)  # [B, To, d]

            # greedy autoregressive rollout of full token budget
            action_tokens = torch.full(
                (B, 1), policy.bos_id, dtype=torch.long, device=device,
            )
            for _ in range(max_k):
                logits = policy.model(action_tokens, cond=features)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                action_tokens = torch.cat([action_tokens, next_token], dim=1)
            action_tokens = action_tokens[:, 1:]  # drop <BOS>

            # RMS per-dim error in [-1, 1] normalized space
            gt_action_norm = action_normalizer.normalize(gt_action[:, :n_action_steps])  # [B, Ta, Da]
            errors = torch.zeros(B, max_k, device=device)
            for k in range(1, max_k + 1):
                tokens_k = action_tokens[:, :k]
                action_pred = policy.action_tokenizer.detokenize(tokens=tokens_k)
                action_pred = action_pred[:, :n_action_steps]
                action_pred_norm = action_normalizer.normalize(action_pred)
                diff = (action_pred_norm - gt_action_norm).reshape(B, -1)
                errors[:, k - 1] = diff.pow(2).mean(dim=-1).sqrt()

            feature_chunks.append(features.cpu().numpy())
            error_chunks.append(errors.cpu().numpy())
            if 'task_uid' in obs_dict:
                task_uid_chunks.append(obs_dict['task_uid'].cpu().numpy())

            n_processed += B
            if max_samples is not None and n_processed >= max_samples:
                break

    features_arr = np.concatenate(feature_chunks, axis=0)
    errors_arr = np.concatenate(error_chunks, axis=0)
    out = {
        'features': features_arr,
        'errors': errors_arr,
    }
    if task_uid_chunks:
        out['task_uids'] = np.concatenate(task_uid_chunks, axis=0)

    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **out)
    print(f"Saved {n_processed} samples to {output}")

    # quick stats
    def print_min_k_stats(errors_arr, epsilon):
        n = len(errors_arr)
        ks = np.arange(1, max_k + 1)

        below = errors_arr < epsilon
        has_below = below.any(axis=1)

        # -1 means fail: even max_k tokens did not reach the threshold
        first_below = np.full(n, -1, dtype=np.int64)
        first_below[has_below] = below[has_below].argmax(axis=1) + 1

        success_min_k = first_below[has_below]
        fail_count = (~has_below).sum()
        success_count = has_below.sum()

        print(f"\nStats for epsilon={epsilon} (normalized RMS per-dim error):")
        print(f"  total samples: {n}")
        print(f"  success count: {success_count}")
        print(f"  fail count:    {fail_count}")
        print(f"  fail rate:     {fail_count / n:.3f}")

        if success_count > 0:
            print("\nMin-k stats among successful samples only:")
            print(f"  mean min_k:   {success_min_k.mean():.2f}")
            print(f"  median min_k: {int(np.median(success_min_k))}")

            hist_success = np.bincount(
                success_min_k,
                minlength=max_k + 1,
            )[1:]

            print(f"  histogram success only (k=1..{max_k}): {hist_success}")

            cum_success = np.cumsum(hist_success) / success_count
            print(f"  cumulative success fraction among successful samples:")
            for k, frac in zip(ks, cum_success):
                print(f"    k <= {k}: {frac:.3f}")

            cum_total = np.cumsum(hist_success) / n
            print(f"  cumulative success fraction among all samples:")
            for k, frac in zip(ks, cum_total):
                print(f"    k <= {k}: {frac:.3f}")
        else:
            print("\nNo samples reached the threshold.")

        # Useful for latency estimate: if fail, we would still generate max_k tokens
        effective_k = np.where(has_below, first_below, max_k)
        print("\nEffective-k stats, treating fail as max_k:")
        print(f"  mean effective_k:   {effective_k.mean():.2f}")
        print(f"  median effective_k: {int(np.median(effective_k))}")
        print(
            f"  histogram effective (k=1..{max_k}): "
            f"{np.bincount(effective_k, minlength=max_k + 1)[1:]}"
        )

        print("\nError statistics per prefix length:")
        print("  k | mean_err | median_err | p90_err | p95_err | frac_below_eps")
        for idx, k in enumerate(ks):
            err_k = errors_arr[:, idx]
            print(
                f"  {k:>1} | "
                f"{err_k.mean():.4f}   | "
                f"{np.median(err_k):.4f}     | "
                f"{np.percentile(err_k, 90):.4f}  | "
                f"{np.percentile(err_k, 95):.4f}  | "
                f"{(err_k < epsilon).mean():.3f}"
            )

    print_min_k_stats(errors_arr, epsilon)


if __name__ == '__main__':
    main()
