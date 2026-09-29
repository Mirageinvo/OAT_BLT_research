#!/usr/bin/env python3
"""KDPE-OAT offline density-collapse diagnostic + candidate-hash regression (plan sec. 8, 9).

No simulator. For 256-512 validation observations:
  1. encode obs, sample N candidates exactly as predict_action_bon_free (flat BoN),
  2. SHA256 candidate token ids + decoded actions,
  3. run vote / medoid / kdpe over the SAME candidate tensor via OATPolicy._bon_select,
  4. verify candidates unchanged, global RNG untouched by selection, indices in [0,N),
  5. report density statistics and the stop conditions of the plan.

Bandwidth is NEVER tuned on success rate. Scale info printed here is label-free.

Example:
  python scripts/kdpe_offline_diagnostic.py --ckpt my_models/<libero_long>.ckpt \
      --n_obs 256 -d cuda:0 --out output/eval/aamas27_selector_baselines/kdpe_n8/offline_diagnostic.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import click
import hydra
import torch
from torch.utils.data import DataLoader

from oat.policy.base_policy import BasePolicy
from oat.policy.kdpe import kdpe_diagnostics


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _obs_batches(policy, cfg, device, n_obs: int):
    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    loader = DataLoader(dataset.get_validation_dataset(), batch_size=1, shuffle=False)
    ports = policy.get_observation_ports()
    out = []
    for i, batch in enumerate(loader):
        if i >= n_obs:
            break
        out.append({k: batch["obs"][k].to(device) for k in ports})
    if not out:
        raise RuntimeError("empty validation dataset")
    return out


def aggregate(rows: List[Dict[str, Any]], n: int) -> Dict[str, Any]:
    m = len(rows)
    frac = lambda k: sum(bool(r[k]) for r in rows) / m  # noqa: E731
    nonident = [r for r in rows if not r["identical_endpoints"]]
    hist = Counter(r["idx"] for r in rows)
    scores_med = sorted(r["score_median"] for r in rows)
    return {
        "n_observations": m,
        "finite_fraction": frac("finite"),
        "score_min": min(r["score_min"] for r in rows),
        "score_median_of_medians": scores_med[m // 2],
        "score_max": max(r["score_max"] for r in rows),
        "margin_top1_top2_mean": sum(r["margin_top1_top2"] for r in rows) / m,
        "margin_top1_top2_median": sorted(r["margin_top1_top2"] for r in rows)[m // 2],
        "exact_tie_fraction_nonidentical": (
            sum(r["exact_tie"] for r in nonident) / len(nonident) if nonident else 0.0),
        "identical_endpoint_fraction": 1 - len(nonident) / m,
        "collapsed_fraction": frac("collapsed"),
        "flat_scores_fraction": frac("flat_scores_1e10"),
        "selected_index_hist": {str(i): hist.get(i, 0) for i in range(n)},
        "selected_index_max_share": max(hist.values()) / m,
        "selected_index0_share": hist.get(0, 0) / m,
        "mean_off_diag_kernel": sum(r["mean_off_kernel"] for r in rows) / m,
        "mean_pairwise_dist": {
            "pos": sum(r["mean_pos_dist"] for r in rows) / m,
            "rot_rad": sum(r["mean_rot_dist"] for r in rows) / m,
            "grip": sum(r["mean_grip_dist"] for r in rows) / m,
        },
        "mean_pairwise_norm_quad": {
            "pos": sum(r["mean_pos_quad"] for r in rows) / m,
            "rot": sum(r["mean_rot_quad"] for r in rows) / m,
            "grip": sum(r["mean_grip_quad"] for r in rows) / m,
        },
    }


def stop_conditions(agg: Dict[str, Any], n: int, index_ok: bool) -> Dict[str, bool]:
    """True == condition triggered (must NOT launch the full benchmark)."""
    return {
        "nan_or_inf": agg["finite_fraction"] < 1.0,
        "index_out_of_range": not index_ok,
        "collapsed_gt_5pct": agg["collapsed_fraction"] > 0.05,
        "exact_tie_gt_5pct": agg["exact_tie_fraction_nonidentical"] > 0.05,
        "almost_always_index0": agg["selected_index0_share"] > 0.5,
    }


@click.command()
@click.option("--ckpt", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--n_obs", default=256, show_default=True, type=click.IntRange(1, 2048))
@click.option("--bon_n", default=8, show_default=True)
@click.option("--use_k_tokens", default=8, show_default=True)
@click.option("--temperature", default=1.0, show_default=True)
@click.option("--topk", default=10, show_default=True)
@click.option("--n_action_steps", default=None, type=int, help="override canonical R (default: ckpt value)")
@click.option("--kdpe_bandwidth", default=0.05, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--out", default="output/eval/aamas27_selector_baselines/kdpe_n8/offline_diagnostic.json",
              show_default=True)
def main(ckpt, device, n_obs, bon_n, use_k_tokens, temperature, topk, n_action_steps,
         kdpe_bandwidth, seed, out):
    dev = torch.device(device)
    policy, cfg = BasePolicy.from_checkpoint(ckpt, return_configuration=True)
    policy.to(dev).eval()
    if n_action_steps is not None:
        policy.n_action_steps = int(n_action_steps)
    batches = _obs_batches(policy, cfg, dev, n_obs)
    print(f"{len(batches)} observations, N={bon_n} K={use_k_tokens} R(n_action_steps)="
          f"{policy.n_action_steps} T={temperature} topk={topk} bandwidth={kdpe_bandwidth}")

    rows, hashes = [], []
    dis_vote = dis_medoid = 0
    index_ok = True
    hash_ok = True
    rng_ok = True
    with torch.inference_mode():
        for i, obs in enumerate(batches):
            torch.manual_seed(seed * 100003 + i)
            if dev.type == "cuda":
                torch.cuda.manual_seed_all(seed * 100003 + i)
            feats = policy.obs_encoder(obs)                                  # [1,To,d]
            feat_rep = feats.repeat_interleave(bon_n, dim=0)
            tokens = policy._bon_sample(feat_rep, use_k_tokens, temperature, topk, 0.0)
            cand = policy.action_tokenizer.detokenize(tokens, eval_keep_k=[use_k_tokens] * bon_n)
            H = cand.shape[1]
            R = min(policy.n_action_steps, H)
            cand_b = cand.reshape(1, bon_n, H, cand.shape[2])[0]

            h_tok, h_act = _sha(tokens), _sha(cand_b)
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None
            picks = {
                s: policy._bon_select(cand_b, R, s, feats[0], kdpe_bandwidth=kdpe_bandwidth)
                for s in ("vote", "medoid", "kdpe")
            }
            if not torch.equal(cpu_state, torch.get_rng_state()):
                rng_ok = False
            if cuda_state is not None and not torch.equal(cuda_state, torch.cuda.get_rng_state(dev)):
                rng_ok = False
            if _sha(cand_b) != h_act or _sha(tokens) != h_tok:
                hash_ok = False
            index_ok &= all(0 <= v < bon_n for v in picks.values())

            row = kdpe_diagnostics(cand_b, R, kdpe_bandwidth)
            assert row["idx"] == picks["kdpe"], (row["idx"], picks["kdpe"])
            rows.append(row)
            dis_vote += picks["kdpe"] != picks["vote"]
            dis_medoid += picks["kdpe"] != picks["medoid"]
            hashes.append({"i": i, "tokens_sha256": h_tok, "actions_sha256": h_act, **picks})

    agg = aggregate(rows, bon_n)
    agg["disagreement_kdpe_vs_cs_vote"] = dis_vote / len(rows)
    agg["disagreement_kdpe_vs_medoid"] = dis_medoid / len(rows)
    stops = stop_conditions(agg, bon_n, index_ok)
    result = {
        "checkpoint": ckpt,
        "params": {"N": bon_n, "K": use_k_tokens, "R": R, "temperature": temperature, "topk": topk,
                   "bandwidth": kdpe_bandwidth, "bon_prefix_k": 0, "bon_first_temp": 0.0,
                   "seed": seed, "n_obs": len(rows)},
        "candidate_hash_regression": {
            "candidates_unchanged_by_selectors": hash_ok,
            "global_rng_untouched_by_selectors": rng_ok,
            "note": "the same candidate tensor is fed to vote/medoid/kdpe; only selected_idx differs",
        },
        "aggregate": agg,
        "stop_conditions_triggered": stops,
        "PASS": (not any(stops.values())) and hash_ok and rng_ok,
        "per_observation_hashes": hashes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "per_observation_hashes"}, indent=2))
    print(f"\nwrote {out}\nDIAGNOSTIC {'PASS' if result['PASS'] else 'FAIL - do NOT run the full benchmark'}")
    sys.exit(0 if result["PASS"] else 2)


if __name__ == "__main__":
    main()
