#!/usr/bin/env python3
"""Unit / self-tests for the KDPE-OAT endpoint selector (no sim, no checkpoint).

Run either with pytest or directly:  python tests/test_kdpe_selector.py
"""
from __future__ import annotations

import math
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from oat.policy.kdpe import (  # noqa: E402
    kdpe_endpoint_scores,
    kdpe_diagnostics,
    kdpe_endpoint_terms,
    kdpe_select,
    pairwise_rotation_angle,
)

H, R, D = 32, 12, 7
B = 0.05
SIG = (B, 5 * B, 20 * B)


def _cands(n=8, h=H, seed=0, scale=0.02):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, h, D, generator=g) * scale


# ---- 1. N=1 --------------------------------------------------------------------
def test_n1():
    c = _cands(1)
    s = kdpe_endpoint_scores(c, R)
    assert s.shape == (1,) and torch.isfinite(s).all()
    assert kdpe_select(c, R) == 0


# ---- 2. compact group + outlier ------------------------------------------------
def test_outlier_not_selected():
    c = torch.zeros(4, H, D)
    c[:3, R - 1, 0:3] = torch.tensor([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
    c[:3, R - 1, 3:6] = torch.tensor([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
    c[:3, R - 1, 6] = torch.tensor([-1.0, -1.0, -0.95])
    c[3, R - 1] = torch.tensor([0.9, -0.9, 0.9, 1.5, -1.0, 2.0, 1.0])   # far outlier
    s = kdpe_endpoint_scores(c, R)
    idx = int(s.argmax())
    assert idx in (0, 1, 2), (idx, s)
    assert s[3] < s[idx]


# ---- 3. endpoint-only semantics ------------------------------------------------
def test_endpoint_only_semantics():
    base = _cands(8, seed=1)
    a, b = base.clone(), base.clone()
    g = torch.Generator().manual_seed(7)
    a[:, : R - 1] = torch.randn(8, R - 1, D, generator=g)      # differ before endpoint
    b[:, : R - 1] = torch.randn(8, R - 1, D, generator=g) * 3
    a[:, R:] = torch.randn(8, H - R, D, generator=g)           # differ after R
    b[:, R:] = 0.0
    assert torch.equal(a[:, R - 1], b[:, R - 1])
    sa, sb = kdpe_endpoint_scores(a, R), kdpe_endpoint_scores(b, R)
    assert torch.equal(sa, sb)
    assert kdpe_select(a, R) == kdpe_select(b, R)


# ---- 4. endpoint index is exactly R-1 (off-by-one guard) -----------------------
def test_endpoint_index_exact():
    base = _cands(8, seed=2)
    s0 = kdpe_endpoint_scores(base, R)

    for idx_changed, must_change in [(R - 1, True), (R - 2, False), (R, False), (H - 1, False), (0, False)]:
        c = base.clone()
        c[3, idx_changed] += torch.tensor([0.05, -0.04, 0.03, 0.2, -0.2, 0.1, 0.5])
        s = kdpe_endpoint_scores(c, R)
        changed = not torch.equal(s, s0)
        assert changed == must_change, f"index {idx_changed}: changed={changed}, expected {must_change}"


# ---- 5. rotation wrap-around ---------------------------------------------------
def test_rotation_wraparound():
    eps = 0.01
    rv = torch.tensor([[0.0, 0.0, math.pi - eps], [0.0, 0.0, -(math.pi - eps)]])
    ang = pairwise_rotation_angle(rv)
    assert abs(float(ang[0, 1]) - 2 * eps) < 1e-4, ang
    assert float((rv[0] - rv[1]).norm()) > 6.0        # naive Euclid rotvec distance is huge
    # exact +pi vs -pi is the same rotation
    rv2 = torch.tensor([[0.0, 0.0, math.pi], [0.0, 0.0, -math.pi]])
    assert float(pairwise_rotation_angle(rv2)[0, 1]) < 1e-3
    # as a selector: wrap-around pair must form the dense cluster, not the outlier
    c = torch.zeros(3, H, D)
    c[0, R - 1, 3:6] = rv[0]
    c[1, R - 1, 3:6] = rv[1]
    c[2, R - 1, 3:6] = torch.tensor([0.0, 0.0, 0.0])
    s = kdpe_endpoint_scores(c, R)
    assert s[0] > s[2] and s[1] > s[2]


# ---- 6. known rotation angle ---------------------------------------------------
def test_known_rotation_angle():
    c = torch.zeros(2, H, D)
    c[1, R - 1, 3:6] = torch.tensor([0.0, 0.0, math.pi / 2])
    pos_q, rot_q, grip_q = kdpe_endpoint_terms(c, R, B)
    expect = (math.pi / 2 / SIG[1]) ** 2
    assert abs(float(rot_q[0, 1]) - expect) / expect < 1e-4, (float(rot_q[0, 1]), expect)
    assert float(pos_q.abs().max()) == 0.0 and float(grip_q.abs().max()) == 0.0


# ---- 7. gripper component ------------------------------------------------------
def test_gripper_component():
    c = torch.zeros(2, H, D)
    c[1, R - 1, 6] = 1.0
    pos_q, rot_q, grip_q = kdpe_endpoint_terms(c, R, B)
    assert abs(float(grip_q[0, 1]) - (1.0 / SIG[2]) ** 2) < 1e-6      # sigma_grip = 1.0
    s = kdpe_endpoint_scores(c, R, B)
    expected_k = math.exp(-0.5 * 1.0)
    assert abs(float(s[0]) - (1 + expected_k) / 2) < 1e-6


# ---- 8. vectorized vs scalar reference -----------------------------------------
def _reference_scores(cand, R_, b):
    from scipy.spatial.transform import Rotation as Rot
    import numpy as np

    x = cand[:, R_ - 1].double().numpy()
    n = x.shape[0]
    sp, sr, sg = b, 5 * b, 20 * b
    out = []
    for i in range(n):
        acc = 0.0
        for j in range(n):
            dp = np.sum((x[i, :3] - x[j, :3]) ** 2) / sp ** 2
            rel = Rot.from_rotvec(x[i, 3:6]).inv() * Rot.from_rotvec(x[j, 3:6])
            dr = rel.magnitude() ** 2 / sr ** 2
            dg = (x[i, 6] - x[j, 6]) ** 2 / sg ** 2
            acc += math.exp(-0.5 * (dp + dr + dg))
        out.append(acc / n)
    return torch.tensor(out, dtype=torch.float64)


def test_reference_parity():
    try:
        import scipy  # noqa: F401
    except ImportError:
        print("  (skip: scipy missing)")
        return
    for seed in range(5):
        c = _cands(8, seed=10 + seed, scale=0.03)
        c[:, R - 1, 3:6] = torch.randn(8, 3, generator=torch.Generator().manual_seed(seed)) * 0.15
        got = kdpe_endpoint_scores(c, R).double()
        ref = _reference_scores(c, R, B)
        assert torch.allclose(got, ref, rtol=2e-4, atol=1e-6), (got, ref)
        assert int(got.argmax()) == int(ref.argmax())
    # also the 2*arccos(|<q1,q2>|) definition from the task statement
    rv = torch.randn(6, 3, generator=torch.Generator().manual_seed(99)) * 1.2
    from scipy.spatial.transform import Rotation as Rot
    q = torch.from_numpy(Rot.from_rotvec(rv.double().numpy()).as_quat()).double()
    dots = (q @ q.T).abs().clamp(max=1.0)
    ref_ang = 2 * torch.arccos(dots)
    assert torch.allclose(pairwise_rotation_angle(rv).double(), ref_ang, atol=2e-3)


# ---- 9. symmetry, diagonal, finiteness -----------------------------------------
def test_symmetry_and_finite():
    c = _cands(8, seed=3, scale=0.05)
    pos_q, rot_q, grip_q = kdpe_endpoint_terms(c, R)
    q = pos_q + rot_q + grip_q
    assert torch.allclose(q, q.T, atol=1e-4)
    assert float(q.diagonal().abs().max()) < 1e-5
    k = torch.exp(-0.5 * q)
    assert torch.allclose(k.diagonal(), torch.ones(8), atol=1e-6)
    s = kdpe_endpoint_scores(c, R)
    assert torch.isfinite(s).all() and (s > 0).all() and (s <= 1 + 1e-6).all()


# ---- extra: dtype / purity / validation ----------------------------------------
def test_fp16_input_and_no_mutation_no_rng():
    c = _cands(8, seed=4).half()
    snap = c.clone()
    torch.manual_seed(1234)
    state = torch.get_rng_state()
    s = kdpe_endpoint_scores(c, R)
    assert s.dtype == torch.float32 and torch.isfinite(s).all()
    assert torch.equal(c, snap), "candidates mutated in place"
    assert torch.equal(torch.get_rng_state(), state), "global RNG touched"
    assert c.device == snap.device


def test_input_validation():
    c = _cands(4)
    for bad in [lambda: kdpe_endpoint_scores(c[0], R),
                lambda: kdpe_endpoint_scores(c[..., :6], R),
                lambda: kdpe_endpoint_scores(c, 0),
                lambda: kdpe_endpoint_scores(c, H + 1),
                lambda: kdpe_endpoint_scores(c, R, bandwidth=0.0)]:
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_tie_break_first_index():
    c = torch.zeros(5, H, D)              # all identical -> equal scores -> index 0
    assert kdpe_select(c, R) == 0


# ---- 10. vote/medoid regression + kdpe wiring in OATPolicy._bon_select ---------
class _IdNorm:
    def normalize(self, x):
        return x


def _stub_policy():
    stub = types.SimpleNamespace()
    stub.action_tokenizer = types.SimpleNamespace(normalizer={"action": _IdNorm()})
    return stub


def _ref_vote(cand, R_):
    flat = cand[:, :R_].reshape(cand.shape[0], -1)
    dist = torch.cdist(flat, flat)
    off = dist[dist > 0]
    sigma = (off.median() if off.numel() > 0 else dist.new_tensor(1.0)) + 1e-6
    return int(torch.exp(-(dist ** 2) / (2 * sigma ** 2)).sum(dim=1).argmax())


def _ref_medoid(cand, R_):
    flat = cand[:, :R_].reshape(cand.shape[0], -1)
    return int(torch.cdist(flat, flat).sum(dim=1).argmin())


def test_bon_select_wiring_and_vote_regression():
    from oat.policy.oatpolicy import OATPolicy

    sel = OATPolicy._bon_select
    stub = _stub_policy()
    for seed in range(20):
        c = _cands(8, seed=100 + seed, scale=0.1)
        assert sel(stub, c, R, "vote") == _ref_vote(c, R)          # CS unchanged
        assert sel(stub, c, R, "medoid") == _ref_medoid(c, R)
        assert sel(stub, c, R, "kdpe", kdpe_bandwidth=B) == kdpe_select(c, R, B)

    # kdpe must not use the OAT normalizer (raw actions)
    class _Boom:
        def normalize(self, x):
            raise AssertionError("normalizer used by kdpe")

    boom = types.SimpleNamespace(action_tokenizer=types.SimpleNamespace(normalizer={"action": _Boom()}))
    c = _cands(8, seed=5)
    assert 0 <= sel(boom, c, R, "kdpe") < 8

    # selection over candidates leaves them untouched and picks a valid index
    snap = c.clone()
    for sig in ("vote", "medoid", "kdpe"):
        i = sel(stub, c, R, sig)
        assert 0 <= i < 8
    assert torch.equal(c, snap)


def test_kdpe_differs_from_cs_by_construction():
    """Prefix agreement vs endpoint agreement: candidates 4..7 share an identical prefix
    (CS-dense) but scatter at the endpoint; candidates 0..3 have wild prefixes but a tight
    endpoint cluster. CS must pick from 4..7, KDPE from 0..3."""
    from oat.policy.oatpolicy import OATPolicy

    stub = _stub_policy()
    g = torch.Generator().manual_seed(0)
    c = torch.zeros(8, H, D)
    c[:4, : R - 1] = torch.randn(4, R - 1, D, generator=g)                # wild prefixes
    c[:4, R - 1] = torch.randn(4, D, generator=g) * 0.002                 # tight endpoint cluster
    c[4:, R - 1, 0:3] = torch.tensor([[0.3, 0, 0], [-0.3, 0, 0], [0, 0.3, 0], [0, -0.3, 0]])
    vote = OATPolicy._bon_select(stub, c, R, "vote")
    kdpe = OATPolicy._bon_select(stub, c, R, "kdpe")
    assert vote in (4, 5, 6, 7), vote
    assert kdpe in (0, 1, 2, 3), kdpe


def test_diagnostics_detect_collapse_and_healthy():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kdpe_offline_diagnostic",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "kdpe_offline_diagnostic.py"))
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)

    g = torch.Generator().manual_seed(0)
    healthy, collapsed = [], []
    for _ in range(64):
        c = torch.zeros(8, H, D)
        c[:, R - 1, :3] = torch.randn(8, 3, generator=g) * 0.03        # spread << sigma_pos
        c[:, R - 1, 6] = torch.randn(8, generator=g) * 0.05
        healthy.append(kdpe_diagnostics(c, R))
        c2 = c.clone()
        c2[:, R - 1, :3] = torch.randn(8, 3, generator=g) * 5.0        # spread >> sigma_pos
        collapsed.append(kdpe_diagnostics(c2, R))
    ah, ac = diag.aggregate(healthy, 8), diag.aggregate(collapsed, 8)
    assert ah["collapsed_fraction"] == 0.0 and ah["finite_fraction"] == 1.0
    assert ac["collapsed_fraction"] > 0.9
    assert not any(diag.stop_conditions(ah, 8, True).values())
    stops = diag.stop_conditions(ac, 8, True)
    assert stops["collapsed_gt_5pct"] and stops["almost_always_index0"], (stops, ac)


def _fake_policy(cands_fixed):
    """Minimal OATPolicy stand-in exercising the REAL predict_action_adaptive/bon_free code."""
    from oat.policy.oatpolicy import OATPolicy

    calls = {"sample": 0, "detok": 0}
    pol = types.SimpleNamespace()
    pol.device = torch.device("cpu")
    pol.max_seq_len = 8
    pol.temperature, pol.topk = 1.0, 10
    pol.n_action_steps = R
    pol.bos_id = 0
    pol.action_tokenizer = types.SimpleNamespace(normalizer={"action": _IdNorm()})
    pol.obs_encoder = lambda obs: torch.zeros(1, 2, 4)

    def _bon_sample(cond, n_new, temperature, topk, first_temp, return_logprobs=False):
        calls["sample"] += 1
        return torch.zeros(cond.shape[0], n_new, dtype=torch.long)

    def detokenize(tokens, eval_keep_k=None):
        calls["detok"] += 1
        return cands_fixed.clone()

    pol._bon_sample = _bon_sample
    pol.action_tokenizer.detokenize = detokenize
    pol._bon_select = types.MethodType(OATPolicy._bon_select, pol)
    pol.predict_action_bon_free = types.MethodType(OATPolicy.predict_action_bon_free, pol)
    pol.predict_action_adaptive = types.MethodType(OATPolicy.predict_action_adaptive, pol)
    return pol, calls


def test_policy_path_plumbing_and_shared_candidates():
    c = _cands(8, seed=21, scale=0.05)
    kw = dict(use_k_tokens=8, temperature=1.0, topk=10, bon_free=8)
    pol_v, calls_v = _fake_policy(c)
    pol_k, calls_k = _fake_policy(c)
    rv = pol_v.predict_action_adaptive({}, bon_signal="vote", **kw)
    rk = pol_k.predict_action_adaptive({}, bon_signal="kdpe", kdpe_bandwidth=B, **kw)
    assert calls_v == calls_k == {"sample": 1, "detok": 1}       # one forward, same generation path
    assert rk["action"].shape == (1, R, D)
    assert int(rk["selected_idx"]) == kdpe_select(c, R, B)
    assert int(rv["selected_idx"]) == _ref_vote(c, R)
    # the executed chunk really is the selected candidate
    assert torch.equal(rk["action_pred"][0], c[int(rk["selected_idx"])])
    # kdpe_bandwidth is actually plumbed to the selector: a huge bandwidth flattens scores
    # differently from a tiny one on the same candidates for at least one seed
    seen = set()
    for seed in range(30):
        cc = _cands(8, seed=300 + seed, scale=0.08)
        pol, _ = _fake_policy(cc)
        i_small = int(pol.predict_action_adaptive({}, bon_signal="kdpe", kdpe_bandwidth=0.02, **kw)["selected_idx"])
        i_large = int(pol.predict_action_adaptive({}, bon_signal="kdpe", kdpe_bandwidth=5.0, **kw)["selected_idx"])
        assert i_small == kdpe_select(cc, R, 0.02) and i_large == kdpe_select(cc, R, 5.0)
        seen.add(i_small != i_large)
    assert True in seen, "kdpe_bandwidth has no effect -> not plumbed"


def test_kdpe_rejects_non_flat_bon():
    pol, _ = _fake_policy(_cands(8))
    for extra in (dict(bon_prefix_k=4), dict(bon_first_temp=1.5)):
        try:
            pol.predict_action_adaptive({}, bon_free=8, bon_signal="kdpe", use_k_tokens=8, **extra)
        except ValueError:
            continue
        raise AssertionError("kdpe must refuse prefix / first-token modes")


TESTS = [
    test_n1, test_outlier_not_selected, test_endpoint_only_semantics, test_endpoint_index_exact,
    test_rotation_wraparound, test_known_rotation_angle, test_gripper_component,
    test_reference_parity, test_symmetry_and_finite, test_fp16_input_and_no_mutation_no_rng,
    test_input_validation, test_tie_break_first_index,
    test_bon_select_wiring_and_vote_regression, test_kdpe_differs_from_cs_by_construction,
    test_diagnostics_detect_collapse_and_healthy,
    test_policy_path_plumbing_and_shared_candidates, test_kdpe_rejects_non_flat_bon,
]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
