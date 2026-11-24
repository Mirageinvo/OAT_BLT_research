"""Is LiberoEnv.reset(seed=s) deterministic regardless of what the env did before? (needs sim)

  MUJOCO_GL=egl .venv/bin/python scripts/rebuttal/diag_reset_determinism.py \
      --manifest eval_out/rebuttal_cs_ablation/main/base/manifest.json --n_tasks 3

For each task: reset(seed) -> state A; reset(seed) again -> B (no history);
then run 200 steps closing the gripper -> reset(seed) -> C; run 200 steps opening -> reset -> D;
200 random steps -> reset -> E. Compares B..E to A per joint group (robot arm / gripper / objects)
and qvel, and prints the scene hash the runner would compute. Also prints gripper/controller
attributes that might carry over between episodes.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from oat.env.libero.env import LiberoEnv  # noqa: E402


def raw(env):
    return env.env.env        # LiberoEnv.env = ControlEnv; ControlEnv.env = robosuite env


def joint_groups(sim):
    m = sim.model
    names = [m.joint_id2name(j) for j in range(m.njnt)]
    adr = m.jnt_qposadr
    groups = {}
    for j, n in enumerate(names):
        a = int(adr[j])
        b = int(adr[j + 1]) if j + 1 < m.njnt else m.nq
        key = 'arm' if n.startswith('robot0_joint') else 'gripper' if 'finger' in n or 'gripper' in n else 'objects'
        groups.setdefault(key, []).extend(range(a, b))
    return groups


def snapshot(env):
    sim = raw(env).sim
    st = np.asarray(env.env.get_sim_state(), dtype=np.float64)
    h = hashlib.md5(np.round(st, 8).tobytes()).hexdigest()[:16]
    return dict(qpos=sim.data.qpos.copy(), qvel=sim.data.qvel.copy(), hash=h)


def carry(env):
    r = raw(env).robots[0]
    out = {}
    for name, obj in [('gripper', getattr(r, 'gripper', None)), ('controller', getattr(r, 'controller', None))]:
        for attr in ['current_action', 'goal_pos', 'goal_ori', 'torques']:
            v = getattr(obj, attr, None) if obj is not None else None
            if v is not None:
                out[f'{name}.{attr}'] = np.round(np.asarray(v, dtype=float).ravel()[:4], 4).tolist()
    return out


def run(env, n, grip, rng):
    for _ in range(n):
        a = np.zeros(7)
        if grip == 'random':
            a = rng.uniform(-1, 1, 7)
        else:
            a[6] = grip
        env.step(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--n_tasks', type=int, default=3)
    ap.add_argument('--steps', type=int, default=200)
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    names, seeds = man['env_task_names'], man['env_seeds']
    seen, picks = set(), []
    for t, s in zip(names, seeds):
        if t not in seen:
            seen.add(t)
            picks.append((t, int(s)))
    rng = np.random.default_rng(0)
    for task, seed in picks[:a.n_tasks]:
        env = LiberoEnv(task)
        env.reset(seed=seed)
        A = snapshot(env)
        groups = joint_groups(raw(env).sim)
        print(f"\n== {task} seed={seed}  hash A={A['hash']}  groups: { {k: len(v) for k, v in groups.items()} }")
        print("   carry-over attrs after reset A:", carry(env))
        cases = [('B: reset again', None), ('C: after gripper close', 1.0),
                 ('D: after gripper open', -1.0), ('E: after random actions', 'random')]
        for label, grip in cases:
            if grip is not None:
                run(env, a.steps, grip, rng)
                print(f"   before reset {label[0]}:", carry(env))
            env.reset(seed=seed)
            X = snapshot(env)
            diffs = {k: float(np.abs(X['qpos'][idx] - A['qpos'][idx]).max()) for k, idx in groups.items()}
            print(f"   {label:26s} hash={X['hash']} same={X['hash'] == A['hash']}  "
                  f"max|dqpos| " + " ".join(f"{k}={v:.2e}" for k, v in diffs.items()) +
                  f"  max|dqvel|={float(np.abs(X['qvel'] - A['qvel']).max()):.2e}")
        env.close()
        env2 = LiberoEnv(task)          # fresh env object, as when a worker switches task
        env2.reset()                    # init_fn does an unseeded reset first
        env2.reset(seed=seed)
        X = snapshot(env2)
        diffs = {k: float(np.abs(X['qpos'][idx] - A['qpos'][idx]).max()) for k, idx in groups.items()}
        print(f"   {'F: fresh env object':26s} hash={X['hash']} same={X['hash'] == A['hash']}  "
              f"max|dqpos| " + " ".join(f"{k}={v:.2e}" for k, v in diffs.items()))
        env2.close()


if __name__ == '__main__':
    main()
