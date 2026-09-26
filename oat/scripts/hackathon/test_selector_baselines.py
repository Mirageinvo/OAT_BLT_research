#!/usr/bin/env python3
"""Unit tests for BoN selector logic (no sim)."""

from __future__ import annotations

import torch


def _select_medoid(cand: torch.Tensor, R: int) -> int:
    flat = cand[:, :R].reshape(cand.shape[0], -1)
    dist = torch.cdist(flat, flat)
    return int(dist.sum(dim=1).argmin())


def _select_vote(cand: torch.Tensor, R: int) -> int:
    flat = cand[:, :R].reshape(cand.shape[0], -1)
    dist = torch.cdist(flat, flat)
    off = dist[dist > 0]
    sigma = (off.median() if off.numel() > 0 else dist.new_tensor(1.0)) + 1e-6
    dens = torch.exp(-(dist ** 2) / (2 * sigma ** 2)).sum(dim=1)
    return int(dens.argmax())


def _select_max_likelihood(seq_lp: torch.Tensor) -> int:
    return int(seq_lp.argmax().item())


def _select_random(N: int, gen: torch.Generator) -> int:
    if N == 1:
        return 0
    return int(torch.randint(0, N, (1,), generator=gen).item())


def test_n1_all_return_zero():
    cand = torch.randn(1, 16, 7)
    assert _select_medoid(cand, 16) == 0
    assert _select_vote(cand, 16) == 0
    assert _select_max_likelihood(torch.tensor([-1.0])) == 0
    g = torch.Generator().manual_seed(0)
    assert _select_random(1, g) == 0


def test_medoid_picks_central_cluster():
    # 4 points near origin, 1 outlier far away -> medoid in the cluster
    R, D = 4, 2
    cluster = torch.randn(4, R, D) * 0.05
    outlier = torch.ones(1, R, D) * 10.0
    cand = torch.cat([cluster, outlier], dim=0)
    idx = _select_medoid(cand, R)
    assert idx in {0, 1, 2, 3}, f"medoid picked outlier idx={idx}"


def test_max_likelihood_known_argmax():
    seq_lp = torch.tensor([-10.0, -1.0, -5.0, -1.0])  # tie at -1 -> lowest index = 1
    assert _select_max_likelihood(seq_lp) == 1


def test_topk_masked_logprob_matches_sampling_dist():
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.1, -5.0]])
    temperature = 1.0
    top_k = 2
    scaled = logits / temperature
    v, _ = torch.topk(scaled, top_k)
    masked = scaled.clone()
    masked[masked < v[:, [-1]]] = -float("Inf")
    log_probs = torch.log_softmax(masked, dim=-1)
    # tokens outside top-2 must be -inf
    assert torch.isneginf(log_probs[0, 0]) or log_probs[0, 0] < -50
    assert torch.isneginf(log_probs[0, 3])
    assert torch.isneginf(log_probs[0, 4])
    # top-2 (indices 1,2 for values 2 and 3) finite and sum to 1 in prob space
    probs = log_probs.exp()
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_random_reproducible_and_near_uniform():
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    seq1 = [_select_random(8, g1) for _ in range(200)]
    seq2 = [_select_random(8, g2) for _ in range(200)]
    assert seq1 == seq2
    counts = torch.zeros(8)
    for i in seq1:
        counts[i] += 1
        assert 0 <= i < 8
    # chi-ish: each bin roughly 200/8=25; allow wide band
    assert counts.min() >= 5
    assert counts.max() <= 60


def test_vote_regression_fixed_candidates():
    torch.manual_seed(0)
    cand = torch.randn(8, 16, 7)
    # denser cluster around first 3
    cand[:3] = cand[:3] * 0.01 + 0.5
    cand[3:] = cand[3:] + 3.0
    idx = _select_vote(cand, 16)
    assert 0 <= idx < 8
    # recompute — deterministic
    assert _select_vote(cand, 16) == idx


def test_no_nan_distances():
    cand = torch.randn(5, 8, 7)
    flat = cand.reshape(5, -1)
    dist = torch.cdist(flat, flat)
    assert torch.isfinite(dist).all()


if __name__ == "__main__":
    tests = [
        test_n1_all_return_zero,
        test_medoid_picks_central_cluster,
        test_max_likelihood_known_argmax,
        test_topk_masked_logprob_matches_sampling_dist,
        test_random_reproducible_and_near_uniform,
        test_vote_regression_fixed_candidates,
        test_no_nan_distances,
    ]
    for t in tests:
        t()
        print(f"OK {t.__name__}")
    print("ALL PASSED")
