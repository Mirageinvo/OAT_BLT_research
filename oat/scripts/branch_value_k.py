"""
PILOT (existence test) for C1:  value(k) — does the marginal value of action-token
fidelity concentrate at CONTACT/critical states (value), NOT at fast-motion (where
reconstruction-hardness lives)?  Counterfactual branching: hold the sim state fixed,
vary only the token budget k, roll the episode to the end, measure success.

Per branch state s (visited by the full k=8 policy):
  - generate ONE set of 8 tokens at s
  - decode the SAME tokens at k=1 (coarse) and k=8 (full)  -> A1, A8   (prefixes)
  - for each: restore s, execute the chunk open-loop for R steps, then CONTINUE with
    the full k=8 policy to episode end; repeat M times -> success rate p1, p8
  - gap(s) = p8 - p1                     (marginal value of fidelity AT s)
  - phase tags: grip_will_change (grasp/release imminent), eef_vel, gripper openness
  - recon proxy:  ||norm(A8) - norm(A1)|| over the executed R steps (reconstruction gap)

PILOT design = maximize signal: k in {1,8}, large R (open-loop compounds the k=1 error),
contact-focused sampling, M continuations.  Answers "is there ANY gap, and is it at
contact?" + calibrates the confirmatory run.

Run on the sim machine:
  cd oat && MUJOCO_GL=egl uv run python scripts/branch_value_k.py \
      -c my_models/policy_ep-0250_sr-0.596.ckpt -o my_datasets/branch_value_k_pilot.npz \
      --n_branch 80 --M 5 --R 32 --k_coarse 1
"""
if __name__ == "__main__":
    import sys, os, pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import json
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


def continue_to_end(env, policy, obs_deque, ports, n_obs, n_act, device, dtype, already_success):
    """Replan with the full k=8 policy until the episode ends. Returns success bool."""
    succ = already_success
    while not env.done and env.cur_step < env.max_episode_steps:
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


def estimate_success(env, ctrl, snap, snap_step, branch_deque, chunk, R, M,
                     policy, ports, n_obs, n_act, device, dtype):
    """Mean success over M continuations of: restore -> exec chunk[:R] open-loop -> continue k=8."""
    succs = []
    for _ in range(M):
        restore(env, ctrl, snap, snap_step)
        dq = deque(copy.deepcopy(list(branch_deque)), maxlen=n_obs + 1)
        succ = False
        for t in range(R):                      # open-loop execute the k-chunk
            if env.done:
                break
            obs, r, done, _, _ = env.step(chunk[t])
            dq.append(obs)
            succ = succ or (r >= 1)
        succ = continue_to_end(env, policy, dq, ports, n_obs, n_act, device, dtype, succ)
        succs.append(float(succ))
    return float(np.mean(succs))


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


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output', required=True, help='output .npz of per-state rows')
@click.option('-d', '--device', default='cuda:0')
@click.option('--n_branch', default=80, type=int, help='target number of branch states')
@click.option('--M', 'M', default=5, type=int, help='continuations per (state,k)')
@click.option('--R', 'R', default=16, type=int,
              help='open-loop steps the k-chunk is executed (16 keeps p_full off the floor; '
                   'R=32 over-floors success at grasp states and masks the gap)')
@click.option('--k_coarse', default=1, type=int, help='coarse budget (vs full k=8)')
@click.option('--n_tasks', default=2, type=int, help='how many libero10 tasks to sweep')
@click.option('--free_frac', default=0.35, type=float,
              help='prob of branching at a non-contact (free-motion) state, for the control stratum')
@click.option('--seed', default=0, type=int)
def main(checkpoint, output, device, n_branch, M, R, k_coarse, n_tasks, free_frac, seed):
    device = torch.device(device)
    rng = np.random.RandomState(seed)

    policy, cfg = BasePolicy.from_checkpoint(checkpoint, return_configuration=True)
    assert isinstance(policy, OATPolicy)
    policy.to(device).eval()
    dtype = policy.dtype
    ports = policy.get_observation_ports()
    n_obs = policy.n_obs_steps
    n_act = policy.n_action_steps
    K = policy.max_seq_len
    norm = policy.action_tokenizer.normalizer['action']
    grip_dim = -1  # last action dim = gripper
    ekw = env_kwargs_from_cfg(cfg)
    print(f"ports={ports} n_obs={n_obs} n_act={n_act} K={K} R={R} M={M} k_coarse={k_coarse}")

    tasks = get_subtasks('libero10')[:n_tasks]
    rows = []  # each: dict of scalars

    pbar = tqdm.tqdm(total=n_branch, desc='branch states')
    for task in tasks:
        if len(rows) >= n_branch:
            break
        env = LiberoEnv(task_name=task, **ekw)
        # never auto-terminate on robosuite horizon — we control termination via
        # LiberoEnv.done (success / max_episode_steps); branching re-steps the env a lot.
        try:
            env.env.env.ignore_done = True
        except Exception:
            pass
        try:
            while len(rows) < n_branch:
                obs, _ = env.reset()
                obs_deque = deque([obs], maxlen=n_obs + 1)
                ref_succ = False
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
                    A8 = decode_k(policy, tokens, K)
                    Ac = decode_k(policy, tokens, k_coarse)

                    # SUSTAINED gripper-command transition (open<->close) = grasp/release
                    # imminent. (naive sign-flip-anywhere fires on gripper-channel noise.)
                    g0 = float(np.mean(A8[:4, grip_dim]))
                    g1 = float(np.mean(A8[max(0, R - 4):R, grip_dim]))
                    grip_will_change = bool(np.sign(g0) != np.sign(g1) and abs(g0) > 0.5 and abs(g1) > 0.5)
                    ncon = int(getattr(ctrl.env.sim.data, 'ncon', 0))  # active MuJoCo contacts

                    branch_deque = deque(copy.deepcopy(list(obs_deque)), maxlen=n_obs + 1)

                    take = grip_will_change or (rng.rand() < free_frac)
                    if take:
                        # recon proxy: ||norm(A8)-norm(Ac)|| over executed R steps
                        with torch.inference_mode():
                            a8n = norm.normalize(torch.from_numpy(A8[:R]).to(device, dtype))
                            acn = norm.normalize(torch.from_numpy(Ac[:R]).to(device, dtype))
                            recon_gap = float((a8n - acn).norm(dim=-1).mean().item())

                        p8 = estimate_success(env, ctrl, snap, snap_step, branch_deque,
                                              A8, R, M, policy, ports, n_obs, n_act, device, dtype)
                        pc = estimate_success(env, ctrl, snap, snap_step, branch_deque,
                                              Ac, R, M, policy, ports, n_obs, n_act, device, dtype)
                        rows.append(dict(
                            task=task, step=int(snap_step),
                            p_full=p8, p_coarse=pc, gap=p8 - pc,
                            grip_will_change=int(grip_will_change), ncon=ncon,
                            eef_vel=eef_vel, gripper_open=gripper_open,
                            recon_gap=recon_gap,
                        ))
                        pbar.update(1)
                        # restore to the reference state to continue the reference rollout
                        obs = restore(env, ctrl, snap, snap_step)
                        obs_deque = deque(copy.deepcopy(list(branch_deque)), maxlen=n_obs + 1)

                    # advance the reference rollout with the full k=8 chunk
                    for t in range(n_act):
                        if env.done:
                            break
                        obs, r, done, _, _ = env.step(A8[t])
                        obs_deque.append(obs)
                        ref_succ = ref_succ or (r >= 1)
        finally:
            env.close()
    pbar.close()

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
    gap = arrs['gap']; gwc = arrs['grip_will_change'].astype(bool)
    rec = arrs['recon_gap']; vel = arrs['eef_vel']
    print(f"\n=== PILOT RESULT (n={len(rows)}, M={M}, R={R}, k {k_coarse} vs {K}) ===")
    print(f"overall: p_full={arrs['p_full'].mean():.3f}  p_coarse={arrs['p_coarse'].mean():.3f}  "
          f"gap={gap.mean():.3f} +/- {gap.std(ddof=1)/np.sqrt(len(gap)):.3f}")

    def stratum(mask, name):
        if mask.sum() == 0:
            print(f"  [{name}] n=0"); return
        g = gap[mask]
        se = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else float('nan')
        print(f"  [{name:<14}] n={mask.sum():>3}  gap={g.mean():+.3f} +/- {se:.3f}  "
              f"(p_full={arrs['p_full'][mask].mean():.3f} p_coarse={arrs['p_coarse'][mask].mean():.3f})")

    print("\nby phase (C1: gap should be LARGER where grip changes / contact):")
    stratum(gwc, 'grip_change')
    stratum(~gwc, 'no_grip_change')
    vmed = np.median(vel)
    stratum(vel < vmed, 'slow_eef')      # contact-ish (low velocity)
    stratum(vel >= vmed, 'fast_eef')     # fast motion (where reconstruction is hard)
    if 'ncon' in arrs:
        nc = arrs['ncon']; nmed = np.median(nc)
        stratum(nc > nmed, 'high_contact')   # more active MuJoCo contacts
        stratum(nc <= nmed, 'low_contact')
    print(f"\nphase coverage: grip_change={int(gwc.sum())}/{len(gwc)}  "
          f"p_full range=[{arrs['p_full'].min():.2f},{arrs['p_full'].max():.2f}] "
          f"(want p_full off the floor so a gap can show)")

    # killer check: does value-gap line up with reconstruction-gap?
    if len(gap) > 2 and rec.std() > 0:
        r = np.corrcoef(gap, rec)[0, 1]
        print(f"\nkiller check  corr(value_gap, recon_gap) = {r:+.3f}  "
              f"(near 0 / negative => reconstruction does NOT predict value => C1 headline)")
    print("\nDONE")


if __name__ == '__main__':
    main()
