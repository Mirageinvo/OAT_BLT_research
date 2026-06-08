"""
PILOT (existence test) for C1:  value(k) — does the marginal value of action-token
fidelity concentrate at CONTACT/critical states (value), NOT at fast-motion (where
reconstruction-hardness lives)?  Counterfactual branching: hold the sim state fixed,
vary only the token budget k, roll the episode to the end, measure success.

Per branch state s (visited by the full k=8 policy), measure a 2x2 grid p(k,R) by
counterfactual branching (restore s -> exec chunk[:R] open-loop -> continue k=8 -> success,
averaged over M), for k in {1, 8} and R in {R_small, R_large}:
  - value_R(s) = p(R_small) - p(R_large)   replan-sooner benefit; USER hypothesis: LARGE at
                                           contact (committing a grasp open-loop is brittle)
  - value_k(s) = p(k=8) - p(k=1)           token-fidelity benefit; C1: LARGE at contact
  - phase tags: grip_will_change (grasp/release imminent), eef_vel, ncon, gripper openness
  - recon proxy: ||norm(A8) - norm(A1)|| over R_large (does reconstruction predict value_k?)
Tests BOTH axes (R is likely the live one — global R-sweep: smaller R -> higher SR) and
whether either benefit CONCENTRATES at contact (the prereq for adaptive R/k to beat fixed).

Run on the sim machine (parallel):
  cd oat && MUJOCO_GL=egl uv run python scripts/branch_value_k.py \
      -c my_models/policy_ep-0250_sr-0.596.ckpt -o my_datasets/branch_value_kR_pilot.npz \
      --n_branch 80 --M 5 --R_small 8 --R_large 32 --k_coarse 1 --n_tasks 5 --n_workers 8
"""
import sys, os, pathlib
# add repo root to sys.path at MODULE level (not just under __main__) so spawned
# multiprocessing workers — which re-import this module — can import oat.*
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if __name__ == "__main__":
    os.chdir(ROOT_DIR)

import copy
import json
import math
import pathlib
from collections import deque

import click
import numpy as np
import torch
import tqdm

from oat.policy.base_policy import BasePolicy
from oat.policy.oatpolicy import OATPolicy
from oat.env.libero.env import LiberoEnv
from oat.env.libero.factory import get_subtasks
from oat.gymnasium_util.multistep_wrapper import stack_last_n_obs


def maybe_to_torch(x, device, dtype):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return x


def build_obs(obs_deque, n_obs_steps, ports, device, dtype):
    """deque of raw oat-formatted single-step obs -> {port: tensor[1, To, ...]} for the policy."""
    out = {}
    for port in ports:
        vals = stack_last_n_obs([o[port] for o in obs_deque], n_obs_steps)  # [To, ...]
        out[port] = maybe_to_torch(vals[None], device, dtype)               # [1, To, ...]
    return out


@torch.inference_mode()
def gen_tokens(policy, obs_dict):
    """Generate one set of max_seq_len(=8) action tokens at this obs (sampled)."""
    features = policy.obs_encoder(obs_dict)
    B = features.shape[0]
    bos = torch.full((B, 1), policy.bos_id, dtype=torch.long, device=policy.device)
    tokens = policy.model.generate(
        bos, cond=features, max_new_tokens=policy.max_seq_len,
        temperature=policy.temperature, top_k=policy.topk,
    )[:, 1:]
    return tokens  # [1, 8]


@torch.inference_mode()
def decode_k(policy, tokens, k):
    """Decode the SAME tokens at budget k -> action chunk [H, action_dim] (unnormalized, numpy)."""
    a = policy.action_tokenizer.detokenize(tokens, eval_keep_k=[k] * tokens.shape[0])
    return a[0].detach().cpu().numpy()  # [H=32, 7]


def restore(env, ctrl, snap, snap_step):
    """Restore the sim to a snapshot and return the oat-formatted current obs."""
    raw = ctrl.regenerate_obs_from_state(snap)   # set_state + forward + re-render
    env.cur_step = int(snap_step)
    env.done = False
    # robosuite base-env internal episode flags are NOT in the mujoco state; without
    # resetting them, accumulated timestep across many continuations hits horizon=1000
    # and the next step raises "executing action in terminated episode".
    try:
        ctrl.env.timestep = 0
        ctrl.env.done = False
    except Exception:
        pass
    return env._extract_obs(raw)


def continue_to_end(env, policy, obs_deque, ports, n_obs, n_act, device, dtype,
                    already_success, step_limit):
    """Replan with the full k=8 policy until the episode ends (or step_limit). Returns success."""
    succ = already_success
    while not env.done and env.cur_step < step_limit:
        obs_dict = build_obs(obs_deque, n_obs, ports, device, dtype)
        tokens = gen_tokens(policy, obs_dict)
        a = decode_k(policy, tokens, policy.max_seq_len)  # k=8
        for t in range(n_act):
            if env.done:
                break
            obs, r, done, _, _ = env.step(a[t])
            obs_deque.append(obs)
            succ = succ or (r >= 1)
    return succ


def rollout_outcomes(env, ctrl, snap, snap_step, branch_deque, chunk, R, M,
                     policy, ports, n_obs, n_act, device, dtype, cont_cap=0):
    """M binary outcomes of: restore -> exec chunk[:R] open-loop -> continue k=8 to end.
    cont_cap>0 caps each continuation at snap_step+cont_cap steps (speedup; deflates absolute
    success equally across plans, so SELECTION gaps are ~unbiased)."""
    outs = []
    for _ in range(M):
        restore(env, ctrl, snap, snap_step)
        step_limit = env.max_episode_steps if cont_cap <= 0 \
            else min(env.max_episode_steps, snap_step + cont_cap)
        dq = deque(copy.deepcopy(list(branch_deque)), maxlen=n_obs + 1)
        succ = False
        for t in range(min(R, len(chunk))):     # open-loop execute the chunk (clamp to horizon)
            if env.done or env.cur_step >= step_limit:
                break
            obs, r, done, _, _ = env.step(chunk[t])
            dq.append(obs)
            succ = succ or (r >= 1)
        succ = continue_to_end(env, policy, dq, ports, n_obs, n_act, device, dtype, succ, step_limit)
        outs.append(float(succ))
    return outs


def estimate_success(env, ctrl, snap, snap_step, branch_deque, chunk, R, M,
                     policy, ports, n_obs, n_act, device, dtype, cont_cap=0):
    """Mean success over M continuations (wraps rollout_outcomes)."""
    return float(np.mean(rollout_outcomes(
        env, ctrl, snap, snap_step, branch_deque, chunk, R, M,
        policy, ports, n_obs, n_act, device, dtype, cont_cap=cont_cap)))


def env_kwargs_from_cfg(cfg):
    er = cfg.task.policy.env_runner
    def g(name, default):
        try:
            return er[name]
        except Exception:
            return default
    return dict(
        image_size=g('image_size', 128),
        camera_names=list(g('camera_names', ['agentview', 'robot0_eye_in_hand'])),
        state_ports=list(g('state_ports', ['robot0_joint_pos', 'robot0_eef_pos',
                                           'robot0_eef_quat', 'robot0_gripper_qpos'])),
        max_episode_steps=g('max_episode_steps', 550),
    )


def collect_rows(checkpoint, device, tasks, n_branch, M, R_small, R_large, k_coarse, free_frac,
                 seed, env_seed, bon_n=0, bon_isolate=False, bon_cap=0,
                 temperature=None, topk=None, show_pbar=True):
    """Collect up to n_branch branch-state rows on `tasks` (one sequential env).
    Per state, measures the 2x2 grid p(k,R) for k in {k_coarse, 8}, R in {R_small, R_large}:
      value_R = p(R_small)-p(R_large)  (replan-sooner value; user hypothesis: large at contact)
      value_k = p(k=8)-p(k=1)          (token-fidelity value; C1)
    Self-contained (loads its own policy) so it can run in a multiprocessing worker."""
    torch.set_num_threads(1)   # avoid BLAS thread contention across workers (bottleneck is MuJoCo physics)
    device = torch.device(device)
    rng = np.random.RandomState(seed)
    policy, cfg = BasePolicy.from_checkpoint(checkpoint, return_configuration=True)
    assert isinstance(policy, OATPolicy)
    policy.to(device).eval()
    if temperature is not None:   # inject sampling diversity for the BoN gate
        policy.temperature = temperature
    if topk is not None:
        policy.topk = topk
    dtype = policy.dtype
    ports = policy.get_observation_ports()
    n_obs = policy.n_obs_steps
    n_act = policy.n_action_steps
    K = policy.max_seq_len
    norm = policy.action_tokenizer.normalizer['action']
    grip_dim = -1
    ekw = env_kwargs_from_cfg(cfg)

    rows = []
    pbar = tqdm.tqdm(total=n_branch, desc=f'branch[s{env_seed}]', disable=not show_pbar)
    for task in tasks:
        if len(rows) >= n_branch:
            break
        env = LiberoEnv(task_name=task, seed=env_seed, **ekw)
        try:
            env.env.env.ignore_done = True   # we control termination, not robosuite horizon
        except Exception:
            pass
        try:
            while len(rows) < n_branch:
                obs, _ = env.reset()
                obs_deque = deque([obs], maxlen=n_obs + 1)
                while not env.done and env.cur_step < env.max_episode_steps and len(rows) < n_branch:
                    ctrl = env.env
                    snap = ctrl.get_sim_state().copy()
                    snap_step = env.cur_step
                    eef = np.asarray(obs_deque[-1]['robot0_eef_pos'], dtype=np.float64)
                    eef_prev = np.asarray(obs_deque[-2]['robot0_eef_pos'], dtype=np.float64) \
                        if len(obs_deque) >= 2 else eef
                    eef_vel = float(np.linalg.norm(eef - eef_prev))
                    gripper_open = float(np.sum(obs_deque[-1]['robot0_gripper_qpos']))

                    obs_dict = build_obs(obs_deque, n_obs, ports, device, dtype)
                    tokens = gen_tokens(policy, obs_dict)
                    A8 = decode_k(policy, tokens, K)              # needed for grip flag + reference rollout
                    Ac = None if bon_n > 0 else decode_k(policy, tokens, k_coarse)  # grid-only

                    g0 = float(np.mean(A8[:4, grip_dim]))
                    g1 = float(np.mean(A8[max(0, R_large - 4):R_large, grip_dim]))
                    grip_will_change = bool(np.sign(g0) != np.sign(g1) and abs(g0) > 0.5 and abs(g1) > 0.5)
                    ncon = int(getattr(ctrl.env.sim.data, 'ncon', 0))

                    branch_deque = deque(copy.deepcopy(list(obs_deque)), maxlen=n_obs + 1)

                    take = grip_will_change or (rng.rand() < free_frac)
                    if take:
                        phase = dict(task=task, step=int(snap_step),
                                     grip_will_change=int(grip_will_change), ncon=ncon,
                                     eef_vel=eef_vel, gripper_open=gripper_open)
                        if bon_isolate:
                            # plan-isolated: N candidate plans, each scored by M continuations
                            # (averages out continuation luck) -> realizable verifier ceiling.
                            obs_branch = build_obs(branch_deque, n_obs, ports, device, dtype)
                            Msel = max(1, M // 2)
                            p_all, p_sel, p_eval = [], [], []
                            for _ in range(bon_n):
                                toki = gen_tokens(policy, obs_branch)       # fresh sampled plan
                                chunk_i = decode_k(policy, toki, K)
                                outs = rollout_outcomes(env, ctrl, snap, snap_step, branch_deque,
                                                        chunk_i, n_act, M, policy, ports, n_obs,
                                                        n_act, device, dtype, cont_cap=bon_cap)
                                p_all.append(float(np.mean(outs)))
                                p_sel.append(float(np.mean(outs[:Msel])))
                                p_eval.append(float(np.mean(outs[Msel:])) if Msel < M
                                              else float(np.mean(outs)))
                            best = int(np.argmax(p_sel))                    # pick by sel half
                            rows.append(dict(**phase, bon_n=int(bon_n),
                                             baseline=float(np.mean(p_all)),       # random plan, all M (low-var, slightly biased up)
                                             baseline_eval=float(np.mean(p_eval)),  # random plan on eval half (clean split-baseline)
                                             oracle=float(np.max(p_all)),          # best plan (biased up — upper bound)
                                             heldout=float(p_eval[best])))         # selected plan scored on eval half (unbiased)
                        elif bon_n > 0:
                            # oracle best-of-N (pass@k): N full rollouts from s, count successes.
                            c = 0
                            for _ in range(bon_n):
                                restore(env, ctrl, snap, snap_step)
                                dq = deque(copy.deepcopy(list(branch_deque)), maxlen=n_obs + 1)
                                step_limit = env.max_episode_steps if bon_cap <= 0 \
                                    else min(env.max_episode_steps, snap_step + bon_cap)
                                c += int(continue_to_end(env, policy, dq, ports, n_obs,
                                                         n_act, device, dtype, False, step_limit))
                            rows.append(dict(**phase, n_succ=int(c), bon_n=int(bon_n)))
                        else:
                            with torch.inference_mode():
                                a8n = norm.normalize(torch.from_numpy(A8[:R_large]).to(device, dtype))
                                acn = norm.normalize(torch.from_numpy(Ac[:R_large]).to(device, dtype))
                                recon_gap = float((a8n - acn).norm(dim=-1).mean().item())
                            # 2x2 grid: k in {coarse, full} x R in {small, large}
                            cells = {}
                            for kname, chunk in [('kc', Ac), ('kf', A8)]:
                                for rname, RR in [('rs', R_small), ('rl', R_large)]:
                                    cells[f'p_{kname}_{rname}'] = estimate_success(
                                        env, ctrl, snap, snap_step, branch_deque, chunk, RR, M,
                                        policy, ports, n_obs, n_act, device, dtype)
                            rows.append(dict(**phase, recon_gap=recon_gap, **cells))
                        pbar.update(1)
                        obs = restore(env, ctrl, snap, snap_step)
                        obs_deque = deque(copy.deepcopy(list(branch_deque)), maxlen=n_obs + 1)

                    for t in range(n_act):
                        if env.done:
                            break
                        obs, r, done, _, _ = env.step(A8[t])
                        obs_deque.append(obs)
        finally:
            env.close()
    pbar.close()
    return rows


def _worker(payload):
    return collect_rows(**payload)


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output', required=True, help='output .npz of per-state rows')
@click.option('-d', '--device', default='cuda:0')
@click.option('--n_branch', default=80, type=int, help='target number of branch states')
@click.option('--M', 'M', default=5, type=int, help='continuations per (state,k)')
@click.option('--R_small', 'R_small', default=8, type=int, help='short open-loop horizon (replan sooner)')
@click.option('--R_large', 'R_large', default=32, type=int, help='long open-loop horizon (commit longer)')
@click.option('--k_coarse', default=1, type=int, help='coarse budget (vs full k=8)')
@click.option('--n_tasks', default=2, type=int, help='how many libero10 tasks to sweep')
@click.option('--free_frac', default=0.35, type=float,
              help='prob of branching at a non-contact (free-motion) state, for the control stratum')
@click.option('--seed', default=0, type=int)
@click.option('--bon_n', default=0, type=int,
              help='oracle best-of-N gate (#7): if >0, sample N full rollouts per state, '
                   'count successes -> pass@k curve. pass@N >> pass@1 => selection has headroom '
                   '(verifier worth building). pass@N ~= pass@1 => selection washed by replanning '
                   '(strengthens the compounding negative). Overrides the 2x2 grid.')
@click.option('--bon_isolate', is_flag=True, default=False,
              help='plan-isolated BoN: score each of N plans by M continuations (averages out '
                   'continuation luck) -> REALIZABLE verifier ceiling (held-out). '
                   'baseline=mean plan, oracle=best plan (biased), heldout=unbiased selection.')
@click.option('--bon_cap', default=0, type=int,
              help='cap each BoN continuation at branch_step+bon_cap steps (0=to episode end). '
                   '~200 speeds up; deflates absolute success equally across plans so selection '
                   'gaps stay ~unbiased.')
@click.option('--temperature', default=None, type=float,
              help='override sampling temperature (inject diversity for the BoN gate; sweep '
                   '1.0/1.5/2.0 — if headroom grows with temp, low diversity was the bottleneck)')
@click.option('--topk', default=None, type=int, help='override sampling top-k')
@click.option('--n_workers', default=1, type=int,
              help='parallel worker processes (each = own env+policy, distinct seed). '
                   'Branching is embarrassingly parallel; bottleneck is single-threaded '
                   'MuJoCo physics, so N workers ~= N x speedup until CPU cores saturate. '
                   'Each worker holds a CUDA context — watch GPU mem.')
def main(checkpoint, output, device, n_branch, M, R_small, R_large, k_coarse, n_tasks,
         free_frac, seed, bon_n, bon_isolate, bon_cap, temperature, topk, n_workers):
    if bon_isolate and bon_n <= 0:
        bon_n = 4   # isolate needs N>=1 plans; default to 4 if not given
    tasks = get_subtasks('libero10')[:n_tasks]
    mode = (f"BoN-isolate(N={bon_n},M={M})" if bon_isolate else
            f"BoN(N={bon_n})" if bon_n > 0 else f"grid R={{{R_small},{R_large}}} k {k_coarse}vs8")
    print(f"mode={mode} cap={bon_cap} n_branch={n_branch} temp={temperature} topk={topk} "
          f"n_workers={n_workers} tasks={len(tasks)}")

    if n_workers <= 1:
        rows = collect_rows(checkpoint, device, tasks, n_branch, M, R_small, R_large, k_coarse,
                            free_frac, seed=seed, env_seed=seed, bon_n=bon_n, bon_isolate=bon_isolate,
                            bon_cap=bon_cap, temperature=temperature, topk=topk, show_pbar=True)
    else:
        import multiprocessing as mp
        share = (n_branch + n_workers - 1) // n_workers   # ceil
        payloads = [dict(checkpoint=checkpoint, device=device, tasks=tasks, n_branch=share,
                         M=M, R_small=R_small, R_large=R_large, k_coarse=k_coarse, free_frac=free_frac,
                         seed=seed + w, env_seed=seed + 1 + w, bon_n=bon_n, bon_isolate=bon_isolate,
                         bon_cap=bon_cap, temperature=temperature, topk=topk, show_pbar=(w == 0))
                    for w in range(n_workers)]
        ctx = mp.get_context('spawn')
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_worker, payloads)
        rows = [r for sub in results for r in sub][:n_branch]

    if not rows:
        print("NO branch states collected!")
        return

    # ---- save raw ----
    keys = list(rows[0].keys())
    arrs = {k: np.array([r[k] for r in rows],
                        dtype=object if k == 'task' else np.float64) for k in keys}
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrs)
    json.dump(rows, open(output.replace('.npz', '.json'), 'w'), indent=2)
    print(f"\nsaved {len(rows)} branch states -> {output}")

    # ---- analyze ----
    gwc = arrs['grip_will_change'].astype(bool); vel = arrs['eef_vel']

    def phase_masks():
        masks = [(gwc, 'grip_change'), (~gwc, 'no_grip_change')]
        vmed = np.median(vel)
        masks += [(vel < vmed, 'slow_eef'), (vel >= vmed, 'fast_eef')]
        if 'ncon' in arrs:
            nc = arrs['ncon']; nmed = np.median(nc)
            masks += [(nc > nmed, 'high_contact'), (nc <= nmed, 'low_contact')]
        return masks

    # ---- plan-isolated BoN: realizable verifier ceiling ----
    if 'heldout' in arrs:
        N = int(arrs['bon_n'][0]); base = arrs['baseline']; orac = arrs['oracle']; held = arrs['heldout']
        base_e = arrs['baseline_eval'] if 'baseline_eval' in arrs else base  # clean split-baseline
        print(f"\n=== PLAN-ISOLATED BoN (n={len(rows)} states, N={N} plans x M continuations) ===")
        print(f"baseline_eval(random plan, eval half)={base_e.mean():.3f}  "
              f"heldout(selected plan, eval half)={held.mean():.3f}  "
              f"oracle(best plan, biased)={orac.mean():.3f}")
        print(f"REALIZABLE headroom (heldout - baseline_eval) = {held.mean()-base_e.mean():+.3f}  "
              f"(>>0 => a verifier picking plans can really gain; ~0 => apparent BoN gain was "
              f"continuation luck, verifier can't capture it)")
        print(f"  (secondary: heldout-baseline_allM = {held.mean()-base.mean():+.3f};  "
              f"biased-oracle headroom = {orac.mean()-base.mean():+.3f} — upper bound)")
        print("\nrealizable headroom (heldout - baseline_eval) by phase:")
        for mask, name in phase_masks():
            if mask.sum() == 0:
                print(f"  [{name:<14}] n=0"); continue
            h = (held - base_e)[mask]; se = h.std(ddof=1)/np.sqrt(len(h)) if len(h) > 1 else float('nan')
            print(f"  [{name:<14}] n={int(mask.sum()):>3}  {h.mean():+.3f} +/- {se:.3f}  "
                  f"(base_e={base_e[mask].mean():.3f} held={held[mask].mean():.3f})")
        print("\nDONE")
        return

    # ---- BoN mode: pass@k curve (oracle best-of-N) ----
    if 'n_succ' in arrs:
        N = int(arrs['bon_n'][0]); c = arrs['n_succ'].astype(int)
        def passk(c_arr, k):
            cn = math.comb(N, k)
            return float(np.mean([1.0 - math.comb(N - int(ci), k) / cn for ci in c_arr]))
        print(f"\n=== ORACLE BoN (n={len(rows)} states, N={N} rollouts/state) ===")
        ks = sorted(set(k for k in [1, 2, 4, N] if k <= N))   # k>N => comb(N,k)=0 (div-by-zero)
        print("pass@k (overall):  " + "  ".join(f"@{k}={passk(c, k):.3f}" for k in ks))
        print(f"HEADROOM pass@{N}-pass@1 = {passk(c, N) - passk(c, 1):+.3f}  "
              f"(>>0 => selection has headroom; ~0 => washed by replanning)")

        def bon_phase(mask, name):
            if mask.sum() == 0:
                print(f"  [{name:<14}] n=0"); return
            cc = c[mask]
            print(f"  [{name:<14}] n={int(mask.sum()):>3}  "
                  + "  ".join(f"@{k}={passk(cc, k):.3f}" for k in [1, N])
                  + f"  head={passk(cc, N) - passk(cc, 1):+.3f}")
        print("\npass@1 / pass@N / headroom by phase:")
        bon_phase(gwc, 'grip_change'); bon_phase(~gwc, 'no_grip_change')
        vmed = np.median(vel); bon_phase(vel < vmed, 'slow_eef'); bon_phase(vel >= vmed, 'fast_eef')
        if 'ncon' in arrs:
            nc = arrs['ncon']; nmed = np.median(nc)
            bon_phase(nc > nmed, 'high_contact'); bon_phase(nc <= nmed, 'low_contact')
        print("\nDONE")
        return

    rec = arrs['recon_gap']
    pkc_rs, pkc_rl = arrs['p_kc_rs'], arrs['p_kc_rl']   # coarse k=1, R small/large
    pkf_rs, pkf_rl = arrs['p_kf_rs'], arrs['p_kf_rl']   # full   k=8, R small/large
    # value_R = replan-sooner benefit (R_small vs R_large); value_k = token benefit (k8 vs k1)
    value_R_kf = pkf_rs - pkf_rl    # at full k (USER hypothesis: large at contact)
    value_R_kc = pkc_rs - pkc_rl    # at coarse k
    value_k_rl = pkf_rl - pkc_rl    # token value at long R (C1)
    value_k_rs = pkf_rs - pkc_rs    # token value at short R
    print(f"\n=== PILOT RESULT (n={len(rows)}, M={M}, R={{{R_small},{R_large}}}, k {k_coarse} vs 8) ===")
    print(f"overall p(k,R): kc_rs={pkc_rs.mean():.3f} kc_rl={pkc_rl.mean():.3f} "
          f"kf_rs={pkf_rs.mean():.3f} kf_rl={pkf_rl.mean():.3f}")
    print(f"overall value_R(kf)={value_R_kf.mean():+.3f}  value_k(rl)={value_k_rl.mean():+.3f}")

    def by_phase(q, title):
        print(f"\n{title}:")
        def st(mask, name):
            if mask.sum() == 0:
                print(f"  [{name:<14}] n=0"); return
            v = q[mask]; se = v.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else float('nan')
            print(f"  [{name:<14}] n={int(mask.sum()):>3}  {v.mean():+.3f} +/- {se:.3f}")
        st(gwc, 'grip_change'); st(~gwc, 'no_grip_change')
        vmed = np.median(vel); st(vel < vmed, 'slow_eef'); st(vel >= vmed, 'fast_eef')
        if 'ncon' in arrs:
            nc = arrs['ncon']; nmed = np.median(nc)
            st(nc > nmed, 'high_contact'); st(nc <= nmed, 'low_contact')

    by_phase(value_R_kf, "value_R = p(R_small)-p(R_large) at k=8  (USER: should be LARGER at contact)")
    by_phase(value_k_rl, "value_k = p(k=8)-p(k=1) at R_large  (C1: should be LARGER at contact)")
    print(f"\nphase coverage: grip_change={int(gwc.sum())}/{len(gwc)}  "
          f"p(kf_rs) range=[{pkf_rs.min():.2f},{pkf_rs.max():.2f}] (want off the floor)")

    # killer check: does token value-gap line up with reconstruction-gap?
    gap = value_k_rl
    if len(gap) > 2 and rec.std() > 0 and gap.std() > 0:
        r = np.corrcoef(gap, rec)[0, 1]
        print(f"\nkiller check  corr(value_k, recon_gap) = {r:+.3f}  "
              f"(near 0 / negative => reconstruction does NOT predict value => C1 headline)")
    print("\nDONE")


if __name__ == '__main__':
    main()
