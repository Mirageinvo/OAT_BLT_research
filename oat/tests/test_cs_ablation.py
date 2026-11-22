#!/usr/bin/env python3
"""Checks for the CS ablation selectors (no sim, no checkpoint).

Run:  cd oat && uv run python -m pytest tests/test_cs_ablation.py -q
  or  cd oat && uv run python tests/test_cs_ablation.py
"""
from __future__ import annotations

import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from oat.policy.oatpolicy import OATPolicy  # noqa: E402
from oat.policy.cs_ablation import (  # noqa: E402
    gaussian_consensus, shadow_decisions, prefix_repr, endpoint_repr,
)

N, H, R, D = 8, 32, 16, 7


class _AffineNorm:
    """Stand-in for the frozen 'limits' normalizer: per-dim affine map."""
    def __init__(self, scale, offset):
        self.scale, self.offset = scale, offset

    def normalize(self, x):
        return x * self.scale + self.offset


def _dummy(norm):
    return types.SimpleNamespace(
        action_tokenizer=types.SimpleNamespace(normalizer={'action': norm}))


def _norm(seed=123):
    g = torch.Generator().manual_seed(seed)
    return _AffineNorm(torch.rand(D, generator=g) * 3 + 0.1, torch.randn(D, generator=g))


def _sel(norm, cand, signal, **kw):
    return OATPolicy._bon_select(_dummy(norm), cand, R, signal, **kw)


def _cands(seed, n=N, scale=1.0, modes=1):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(modes, H, D, generator=g)
    which = torch.randint(0, modes, (n,), generator=g)
    return base[which] + scale * 0.3 * torch.randn(n, H, D, generator=g)


def _vote_reference(a):
    """Literal copy of the original 'vote' code (52e55e6) for an independent check."""
    flat = a.reshape(a.shape[0], -1)
    dist = torch.cdist(flat, flat)
    off = dist[dist > 0]
    sigma = (off.median() if off.numel() > 0 else dist.new_tensor(1.0)) + 1e-6
    dens = torch.exp(-(dist ** 2) / (2 * sigma ** 2)).sum(dim=1)
    return int(dens.argmax())


def test_cs_prefix_equals_original_vote_and_shadow():
    norm = _norm()
    for seed in range(300):
        c = _cands(seed, modes=1 + seed % 3)
        a = norm.normalize(c[:, :R])
        v = _sel(norm, c, 'vote')
        assert v == _vote_reference(a)
        sd = shadow_decisions(a, R)
        assert sd['cs_prefix_idx'] == v
        assert sd['medoid_prefix_idx'] == _sel(norm, c, 'medoid')
        assert sd['cs_endpoint_idx'] == _sel(norm, c, 'cs_endpoint')


def test_identical_candidates_tie_to_first_index():
    norm = _norm()
    c = _cands(0)[:1].repeat(N, 1, 1)
    for sig in ('vote', 'medoid', 'cs_endpoint'):
        assert _sel(norm, c, sig) == 0, sig
    assert _sel(norm, c, 'cs_prefix_fixed', cs_fixed_h=1.0) == 0
    sd = shadow_decisions(norm.normalize(c[:, :R]), R, fixed_h=1.0)
    assert abs(sd['h_prefix'] - (1.0 + 1e-6)) < 1e-6          # no positive distance rule (float32)
    assert sd['n_unique_prefix'] == 1 and sd['d_cs_medoid'] == 0.0


def test_duplicates_win_and_first_duplicate_is_chosen():
    norm = _norm()
    c = _cands(7, scale=3.0)
    c[5] = c[2]
    c[6] = c[2]
    assert _sel(norm, c, 'vote') == 2


def test_adaptive_h_is_lower_median():
    a = _norm().normalize(_cands(3)[:, :R])
    _, h, dist = gaussian_consensus(prefix_repr(a, R))
    iu = torch.triu_indices(N, N, 1)
    u = torch.sort(dist[iu[0], iu[1]]).values            # 28 unique distances
    assert torch.isclose(h, u[13] + 1e-6)                  # 14th smallest (lower median)


def test_endpoint_uses_only_step_R_minus_1_full_vector():
    norm = _norm()
    c = _cands(11, scale=2.0)
    i0 = _sel(norm, c, 'cs_endpoint')
    c2 = c.clone()
    g = torch.Generator().manual_seed(99)
    c2[:, :R - 1] += 5 * torch.randn(N, R - 1, D, generator=g)   # perturb earlier steps
    c2[:, R:] += 5 * torch.randn(N, H - R, D, generator=g)       # and non-executed tail
    assert _sel(norm, c2, 'cs_endpoint') == i0
    # changing one coordinate (gripper, last dim) at step R-1 must matter
    a = norm.normalize(c[:, :R])
    assert endpoint_repr(a, R).shape == (N, D)


def test_fixed_h_matches_adaptive_when_equal():
    norm = _norm()
    for seed in range(100):
        c = _cands(seed, modes=2)
        a = norm.normalize(c[:, :R])
        _, h, _ = gaussian_consensus(prefix_repr(a, R))
        assert _sel(norm, c, 'cs_prefix_fixed', cs_fixed_h=float(h)) == _sel(norm, c, 'vote')


def test_fixed_h_requires_value():
    try:
        _sel(_norm(), _cands(0), 'cs_prefix_fixed')
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
