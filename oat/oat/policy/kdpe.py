"""KDPE-style endpoint kernel-density selector for OAT candidates ("KDPE-OAT").

Adaptation of KDPE (arXiv:2508.10511, official code hsp-iit/KDPE commit
db0037d09c098626a3da4b3e464322267eccce5f, diffusion_policy/model/filters/kde.py) to the
OAT / LIBERO interface. Only the *selection score* is implemented; candidate generation
stays with OAT.

Differences from the original (must be stated in the paper as "KDPE-style (OAT adaptation)"):
  * OAT/LIBERO actions are relative 7-D commands [dx, dy, dz, rx, ry, rz, gripper]
    (rotation = axis-angle), the original uses absolute pose (pos + rot6d + gripper).
  * Only the action at the last executed step (index R-1, 0-based) is scored.
  * Published bandwidths are used unchanged: sigma = (b, 5b, 20b) = (0.05, 0.25, 1.0).

Input candidates must already be raw / de-normalized environment actions
(`action_tokenizer.detokenize` output). Do NOT apply the OAT normalizer here.

Pure torch (no pytorch3d): quaternions are real-first (w, x, y, z), matching PyTorch3D.
"""
from __future__ import annotations

from typing import Tuple

import torch

ACTION_DIM = 7
SIGMA_ROT_MULT = 5.0
SIGMA_GRIP_MULT = 20.0


def _axis_angle_to_quaternion(rotvec: torch.Tensor) -> torch.Tensor:
    """[..., 3] axis-angle -> [..., 4] unit quaternion (w, x, y, z)."""
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * angle
    small = angle < 1e-6
    # sin(half)/angle, with Taylor expansion 0.5 - angle^2/48 near zero
    safe_angle = torch.where(small, torch.ones_like(angle), angle)
    scale = torch.where(small, 0.5 - angle.square() / 48.0, torch.sin(half) / safe_angle)
    quat = torch.cat([torch.cos(half), rotvec * scale], dim=-1)
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True)


def _quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of real-first quaternions, broadcasting over leading dims."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def pairwise_rotation_angle(rotvec: torch.Tensor) -> torch.Tensor:
    """Pairwise geodesic angle on SO(3) between axis-angle rotations. [N,3] -> [N,N] in [0, pi].

    angle_ij = 2 * atan2(||vec(q_i q_j^-1)||, |w(q_i q_j^-1)|). Equals 2*arccos(|<q_i,q_j>|)
    but numerically stable for near-identical rotations in float32; |w| makes q and -q
    equivalent, so rotations around +pi / -pi are correctly close.
    """
    quat = _axis_angle_to_quaternion(rotvec)                     # [N,4]
    rel = _quat_multiply(quat[:, None, :], _quat_conjugate(quat)[None, :, :])  # [N,N,4]
    w = rel[..., 0].abs()
    v = torch.linalg.norm(rel[..., 1:], dim=-1)
    return 2.0 * torch.atan2(v, w)


def _validate(candidates: torch.Tensor, execution_horizon: int, bandwidth: float) -> None:
    if candidates.ndim != 3:
        raise ValueError(f"Expected [N,H,D], got {tuple(candidates.shape)}")
    n, horizon, action_dim = candidates.shape
    if n < 1:
        raise ValueError("KDPE requires at least one candidate")
    if action_dim != ACTION_DIM:
        raise ValueError(f"KDPE-OAT LIBERO expects D=7, got D={action_dim}")
    if not 1 <= execution_horizon <= horizon:
        raise ValueError(f"execution_horizon must be in [1,{horizon}], got {execution_horizon}")
    if not bandwidth > 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")


@torch.inference_mode()
def kdpe_endpoint_terms(
    candidates: torch.Tensor,
    execution_horizon: int,
    bandwidth: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalized squared pairwise distances (pos, rot, grip), each [N,N], float32.

    Endpoint = candidates[:, execution_horizon - 1] (last executed action, 0-based).
    """
    _validate(candidates, execution_horizon, bandwidth)
    endpoint = candidates[:, execution_horizon - 1].float()      # [N,7]
    pos, rotvec, grip = endpoint[:, 0:3], endpoint[:, 3:6], endpoint[:, 6:7]

    sigma_pos = float(bandwidth)
    sigma_rot = float(bandwidth) * SIGMA_ROT_MULT
    sigma_grip = float(bandwidth) * SIGMA_GRIP_MULT

    pos_quad = (pos[:, None, :] - pos[None, :, :]).square().sum(-1) / sigma_pos ** 2
    rot_quad = pairwise_rotation_angle(rotvec).square() / sigma_rot ** 2
    grip_quad = (grip[:, None, :] - grip[None, :, :]).square().sum(-1) / sigma_grip ** 2
    return pos_quad, rot_quad, grip_quad


@torch.inference_mode()
def kdpe_endpoint_scores(
    candidates: torch.Tensor,      # [N, H, 7], raw (de-normalized) actions
    execution_horizon: int,
    bandwidth: float = 0.05,
) -> torch.Tensor:                 # [N]
    """KDE density rho_i = mean_j exp(-0.5 * q_ij) at the last executed action. float32."""
    _validate(candidates, execution_horizon, bandwidth)
    n = candidates.shape[0]
    if n == 1:
        return torch.ones(1, device=candidates.device, dtype=torch.float32)
    pos_quad, rot_quad, grip_quad = kdpe_endpoint_terms(candidates, execution_horizon, bandwidth)
    scores = torch.exp(-0.5 * (pos_quad + rot_quad + grip_quad)).mean(dim=-1)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("KDPE produced NaN/Inf scores")
    return scores


@torch.inference_mode()
def kdpe_select(
    candidates: torch.Tensor,
    execution_horizon: int,
    bandwidth: float = 0.05,
) -> int:
    """argmax density; torch.argmax returns the first (lowest) index on ties."""
    return int(kdpe_endpoint_scores(candidates, execution_horizon, bandwidth).argmax().item())


@torch.inference_mode()
def kdpe_diagnostics(
    candidates: torch.Tensor,
    execution_horizon: int,
    bandwidth: float = 0.05,
) -> dict:
    """Label-free per-observation diagnostics for density-collapse checks (plan sec. 9).

    Never used for selection or bandwidth tuning on success rate.
    """
    n = candidates.shape[0]
    endpoint = candidates[:, execution_horizon - 1].float()
    pos_q, rot_q, grip_q = kdpe_endpoint_terms(candidates, execution_horizon, bandwidth)
    q = pos_q + rot_q + grip_q
    kern = torch.exp(-0.5 * q)
    scores = kern.mean(dim=-1)
    off_mask = ~torch.eye(n, dtype=torch.bool, device=q.device)
    off = kern[off_mask] if n > 1 else kern.new_zeros(0)
    top2 = torch.topk(scores, k=min(2, n)).values
    identical = bool((endpoint - endpoint[:1]).abs().max() == 0)
    sigma_pos, sigma_rot, sigma_grip = bandwidth, 5 * bandwidth, 20 * bandwidth
    return {
        "finite": bool(torch.isfinite(scores).all()),
        "idx": int(scores.argmax()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_median": float(scores.median()),
        "margin_top1_top2": float(top2[0] - top2[1]) if n > 1 else 0.0,
        "exact_tie": bool(n > 1 and top2[0] == top2[1]),
        "identical_endpoints": identical,
        "collapsed": bool(n > 1 and (off < 1e-12).all()),
        "flat_scores_1e10": bool(n > 1 and (scores.max() - scores.min()) < 1e-10),
        "mean_off_kernel": float(off.mean()) if n > 1 else 1.0,
        # raw (un-normalized) mean pairwise distances, for scale sanity vs the sigmas
        "mean_pos_dist": float((pos_q[off_mask].sqrt() * sigma_pos).mean()) if n > 1 else 0.0,
        "mean_rot_dist": float((rot_q[off_mask].sqrt() * sigma_rot).mean()) if n > 1 else 0.0,
        "mean_grip_dist": float((grip_q[off_mask].sqrt() * sigma_grip).mean()) if n > 1 else 0.0,
        "mean_pos_quad": float(pos_q[off_mask].mean()) if n > 1 else 0.0,
        "mean_rot_quad": float(rot_q[off_mask].mean()) if n > 1 else 0.0,
        "mean_grip_quad": float(grip_q[off_mask].mean()) if n > 1 else 0.0,
    }
