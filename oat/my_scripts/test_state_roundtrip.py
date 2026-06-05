"""
Verify LIBERO/MuJoCo state snapshot + restore works correctly (prereq for the
counterfactual branching experiment: value(k | phase)).

Tests three things:
  1. get_sim_state() / set_state() round-trips (restored eef == snapshot eef).
  2. DETERMINISTIC replay: restore + replay the SAME action sequence reproduces
     the branch exactly (so a k=2 vs k=8 branch from one state is apples-to-apples).
  3. Camera obs are consistent after restore (regenerate_obs_from_state re-renders).

Run on the sim machine (docker):
  cd oat && MUJOCO_GL=egl uv run python my_scripts/test_state_roundtrip.py
"""
import numpy as np
from oat.env.libero.factory import get_subtasks
from oat.env.libero.env import LiberoEnv


def main():
    tasks = get_subtasks('libero10')
    print(f"libero10 has {len(tasks)} tasks; using: {tasks[0]}")

    env = LiberoEnv(task_name=tasks[0], image_size=128, max_episode_steps=550)
    obs, info = env.reset()
    print("reset OK; obs keys:", list(obs.keys()))

    rng = np.random.RandomState(0)
    def act():
        return np.concatenate([rng.uniform(-0.3, 0.3, 6), [rng.choice([-1., 1.])]]).astype(np.float32)

    # advance to a non-trivial mid-episode state
    for _ in range(15):
        obs, r, done, trunc, info = env.step(act())

    ctrl = env.env  # ControlEnv (libero wrapper)
    snap = ctrl.get_sim_state().copy()
    snap_step = env.cur_step
    eef0 = obs['robot0_eef_pos'].copy()
    print(f"\nsnapshot @ step {snap_step}: eef={eef0}, mj_state_dim={snap.shape}")

    # fixed action sequence to apply on both branches
    acts = [act() for _ in range(8)]

    # --- branch A: step from snapshot ---
    obsA = obs
    for a in acts:
        obsA, *_ = env.step(a)
    eefA = obsA['robot0_eef_pos'].copy()

    # --- RESTORE, then replay the SAME actions -> must match branch A ---
    obsR = ctrl.regenerate_obs_from_state(snap)   # set_state + forward + re-render
    env.cur_step = snap_step
    env.done = False
    eef_restored = np.asarray(obsR['robot0_eef_pos']).copy()
    print(f"after restore eef={eef_restored}")
    print(f"  restore vs snapshot eef  max|d| = {np.abs(eef_restored - eef0).max():.2e}  "
          f"(want ~0 — state restored)")

    obsB = obsR
    for a in acts:
        obsB, *_ = env.step(a)
    eefB = obsB['robot0_eef_pos'].copy()
    dmax = np.abs(eefA - eefB).max()
    print(f"\nbranch A  eef = {eefA}")
    print(f"replay    eef = {eefB}")
    print(f"  replay vs branchA max|d| = {dmax:.3e}  -> "
          f"{'MATCH — deterministic restore OK' if dmax < 1e-3 else 'MISMATCH — controller/RNG state not restored!'}")

    # camera consistency after restore (compare restored render to snapshot-time render)
    if 'agentview_rgb' in obs:
        imgR = np.asarray(obsR['agentview_rgb']).astype(np.int32)
        img0 = np.asarray(obs['agentview_rgb']).astype(np.int32)
        print(f"\nrestored agentview vs snapshot agentview  mean|d|px = {np.abs(imgR - img0).mean():.3f} "
              f"(want small — camera re-rendered consistently)")

    env.close()
    print("\nDONE — if replay MATCHes, branching (counterfactual value(k)) is feasible.")


if __name__ == '__main__':
    main()
