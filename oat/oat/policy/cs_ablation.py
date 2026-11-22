"""Consensus Selection (CS) ablation helpers — AAMAS-27 rebuttal.

All variants receive the SAME candidate tensor (N sampled chunks, decoded once) and only
choose an index. Nothing here changes generation, N, R, or what the env executes.

Variants (names used on the CLI):
  cs_prefix        -> existing `vote` path in OATPolicy._bon_select (UNCHANGED; not routed here)
  cs_endpoint      -> same Gaussian score + same adaptive-bandwidth rule, but distances between
                      the normalized FULL action vector at executed step R (index R-1) only
  medoid_prefix    -> existing `medoid` path (UNCHANGED): argmin_i sum_j d_ij on the prefix
  cs_prefix_fixed  -> same prefix representation/distances as cs_prefix, Gaussian score with a
                      single pre-registered bandwidth h for every call

Numerical details shared with the original `vote` code (kept bit-for-bit):
  * d = torch.cdist(flat, flat) (Euclidean) on the normalized, flattened representation.
  * adaptive h = torch.median(d[d > 0]) + 1e-6, or 1 + 1e-6 if no positive distance.
    NOTE: torch.median returns the LOWER of the two middle values for an even count. With
    N=8 the off-diagonal set has 56 entries (each pair twice), so h is the 14th smallest of
    the 28 unique pairwise distances (not the mean of 14th/15th as numpy.median would give).
  * score_i = sum_j exp(-d_ij^2 / (2 h^2)) (self term included; constant, argmax-neutral).
  * ties: torch.argmax / argmin return the FIRST maximal / minimal index.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

CS_VARIANTS = ('cs_prefix', 'cs_endpoint', 'medoid_prefix', 'cs_prefix_fixed')
# CLI variant name -> internal bon_signal. cs_prefix / medoid_prefix map onto the exact
# pre-existing code paths so the control condition is functionally unchanged.
VARIANT_TO_SIGNAL = {
    'cs_prefix': 'vote',
    'medoid_prefix': 'medoid',
    'cs_endpoint': 'cs_endpoint',
    'cs_prefix_fixed': 'cs_prefix_fixed',
}


def gaussian_consensus(x: torch.Tensor, h: Optional[float] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x: [N, F] representation. Returns (scores [N], h used (0-d tensor), dist [N, N]).
    h=None -> adaptive rule identical to the original `vote` code."""
    dist = torch.cdist(x, x)
    if h is None:
        off = dist[dist > 0]
        sigma = (off.median() if off.numel() > 0 else dist.new_tensor(1.0)) + 1e-6
    else:
        sigma = dist.new_tensor(float(h))
    dens = torch.exp(-(dist ** 2) / (2 * sigma ** 2)).sum(dim=1)
    return dens, sigma, dist


def endpoint_repr(a_norm: torch.Tensor, R: int) -> torch.Tensor:
    """a_norm: [N, >=R, D] normalized actions -> [N, D] full action vector at step R (idx R-1)."""
    return a_norm[:, R - 1, :]


def prefix_repr(a_norm: torch.Tensor, R: int) -> torch.Tensor:
    """a_norm: [N, >=R, D] -> [N, R*D] flattened executed prefix."""
    return a_norm[:, :R, :].reshape(a_norm.shape[0], -1)


def select_cs_endpoint(a_norm: torch.Tensor, R: int) -> int:
    dens, _, _ = gaussian_consensus(endpoint_repr(a_norm, R))
    return int(dens.argmax())


def select_cs_prefix_fixed(a_norm: torch.Tensor, R: int, h: float) -> int:
    dens, _, _ = gaussian_consensus(prefix_repr(a_norm, R), h=h)
    return int(dens.argmax())


@torch.inference_mode()
def shadow_decisions(a_norm: torch.Tensor, R: int, fixed_h: Optional[float] = None) -> dict:
    """What every variant WOULD pick on this one candidate set (diagnostics only; called
    outside the timed region). a_norm: [N, >=R, D] normalized candidates."""
    N = a_norm.shape[0]
    flat = prefix_repr(a_norm, R)
    dens, h, dist = gaussian_consensus(flat)
    cs = int(dens.argmax())
    med = int(dist.sum(dim=1).argmin())
    dens_e, h_e, _ = gaussian_consensus(endpoint_repr(a_norm, R))
    ep = int(dens_e.argmax())
    fx = -1
    if fixed_h is not None and fixed_h > 0:
        fx = int(gaussian_consensus(flat, h=fixed_h)[0].argmax())
    # number of distinct executed prefixes (exact duplicates collapse)
    n_unique = int(torch.unique(flat, dim=0).shape[0])
    return dict(
        cs_prefix_idx=cs, medoid_prefix_idx=med, cs_endpoint_idx=ep, cs_prefix_fixed_idx=fx,
        h_prefix=float(h), h_endpoint=float(h_e),
        d_cs_medoid=float(dist[cs, med]),           # normalized-prefix L2 between the two picks
        d_cs_endpoint=float(dist[cs, ep]),
        n_unique_prefix=n_unique, n_cand=N,
    )
