#!/usr/bin/env python3
"""G0b — RoboCasa success-parity smoke (ROBOCASA.md).

Replay K demos open-loop; wrapper success must match env._check_success each step.
Prefer official human HDF5 (states + actions) so reset is deterministic. Zarr is
checked for Da=12 / key shapes. PASS required before policy train.

  MUJOCO_GL=egl python scripts/robocasa_success_parity.py --task coffee_press_button
  MUJOCO_GL=egl python scripts/robocasa_success_parity.py --task close_drawer -k 5

Uses .venv_robocasa (robosuite 1.5 + robocasa). Do not run under shared .venv.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List, Optional, Tuple

import h5py
import numpy as np
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oat.env.robocasa.dataset_conversion import (  # noqa: E402
    EXPECTED_ACTION_DIM,
    TASK_ID_TO_PASCAL,
)
from oat.env.robocasa.env import RoboCasaEnv  # noqa: E402


def _sorted_demo_keys(data_group) -> List[str]:
    def key_fn(name: str):
        if name.startswith("demo_"):
            suffix = name[len("demo_") :]
            if suffix.isdigit():
                return (0, int(suffix))
        return (1, name)

    return sorted([k for k in data_group.keys() if k.startswith("demo_")], key=key_fn)


def _resolve_hdf5(task: str, hdf5: Optional[str]) -> pathlib.Path:
    if hdf5:
        p = pathlib.Path(hdf5)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    pascal = TASK_ID_TO_PASCAL[task]
    candidates = [
        ROOT / "data" / "robocasa" / "hdf5" / pascal / "human" / "demo_gentex_im128_randcams.hdf5",
        ROOT / "data" / "robocasa" / "hdf5" / pascal / "mg" / "demo_gentex_im128_randcams.hdf5",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "No RoboCasa HDF5 for G0b state-reset. Expected one of:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nOr pass --hdf5 PATH"
    )


def _reset_to(env, initial_state: dict) -> None:
    """Restore MuJoCo state like robocasa.scripts.playback_dataset.reset_to.

    Extra vs upstream (needed on robosuite 1.5):
    - copy ep_meta cam_configs into env._cam_configs *after* reset() (reset
      rebuilds default cams; edit_model_xml rewrites parented cams from
      _cam_configs and would otherwise clobber demo camera poses);
    - force generative_textures=None during edit_model_xml so model_file
      texture paths are kept (if the env were constructed with gentex=100p,
      edit would otherwise call get_random_textures and replace the skin).
    Pixel-perfect L1 vs stored demo frames is not guaranteed (rs 1.4→1.5 /
    asset remaps); G0b gates on success parity + eef traj, not image L1.
    """
    import copy

    import robosuite

    inner = env.env  # robosuite env inside RoboCasaEnv
    if "model" in initial_state:
        ep_meta = {}
        if initial_state.get("ep_meta", None) is not None:
            ep_meta = json.loads(initial_state["ep_meta"])
        if hasattr(inner, "set_ep_meta"):
            inner.set_ep_meta(ep_meta)
        elif hasattr(inner, "set_attrs_from_ep_meta"):
            inner.set_attrs_from_ep_meta(ep_meta)
        inner.reset()
        if ep_meta.get("cam_configs"):
            inner._cam_configs = copy.deepcopy(ep_meta["cam_configs"])
        saved_gt = getattr(inner, "generative_textures", None)
        saved_fx = copy.deepcopy(getattr(inner, "_curr_gen_fixtures", None))
        inner.generative_textures = None
        if hasattr(inner, "_curr_gen_fixtures"):
            inner._curr_gen_fixtures = {}
        xml = inner.edit_model_xml(initial_state["model"])
        inner.generative_textures = saved_gt
        if hasattr(inner, "_curr_gen_fixtures"):
            inner._curr_gen_fixtures = saved_fx
        inner.reset_from_xml_string(xml)
        inner.sim.reset()
    if "states" in initial_state:
        inner.sim.set_state_from_flattened(initial_state["states"])
        inner.sim.forward()
    if hasattr(inner, "update_state"):
        inner.update_state()
    elif hasattr(inner, "update_sites"):
        inner.update_sites()
    env.done = False
    env.cur_step = 0
    _ = robosuite  # keep import used for version side-effects / clarity


def _check_zarr(task: str, zarr_path: pathlib.Path, log) -> None:
    if not zarr_path.is_dir():
        raise FileNotFoundError(f"zarr missing: {zarr_path}")
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"] if "data" in root else root
    act = data["action"]
    log(f"zarr={zarr_path} action shape={act.shape} dtype={act.dtype}")
    if act.shape[-1] != EXPECTED_ACTION_DIM:
        raise RuntimeError(f"zarr action dim {act.shape[-1]} != {EXPECTED_ACTION_DIM}")
    for key in (
        "robot0_agentview_left_rgb",
        "robot0_agentview_right_rgb",
        "robot0_eye_in_hand_rgb",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ):
        if key not in data:
            raise KeyError(f"zarr missing obs key {key}")
    log("zarr obs keys OK")


def replay_demo(
    env: RoboCasaEnv,
    actions: np.ndarray,
    initial_state: dict,
) -> Tuple[bool, int, List[str]]:
    """Open-loop replay. Returns (final_success, n_steps, mismatches)."""
    _reset_to(env, initial_state)
    mismatches: List[str] = []
    success = False
    n = int(actions.shape[0])
    for t in range(n):
        _, reward, done, _, _ = env.step(np.asarray(actions[t], dtype=np.float32))
        wrap_succ = bool(reward >= 1.0) or env._check_success()
        raw_succ = bool(env.env._check_success())
        if wrap_succ != raw_succ:
            mismatches.append(f"step={t} wrap={wrap_succ} raw={raw_succ}")
        success = wrap_succ or raw_succ
        if done and success:
            return True, t + 1, mismatches
    return success, n, mismatches


def main() -> int:
    ap = argparse.ArgumentParser(description="RoboCasa G0b success-parity")
    ap.add_argument(
        "--task",
        required=True,
        choices=list(TASK_ID_TO_PASCAL.keys()),
    )
    ap.add_argument("-k", type=int, default=5, help="number of demos to replay")
    ap.add_argument("--hdf5", type=str, default=None)
    ap.add_argument(
        "--zarr",
        type=str,
        default=None,
        help="default: data/robocasa/<task>_N200.zarr",
    )
    ap.add_argument(
        "--log",
        type=str,
        default=None,
        help="default: logs/robocasa_success_parity_<task>.log",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task = args.task
    log_path = pathlib.Path(args.log or f"logs/robocasa_success_parity_{task}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    zarr_path = pathlib.Path(args.zarr or f"data/robocasa/{task}_N200.zarr")
    log(f"=== G0b success-parity | task={task} | k={args.k} ===")
    try:
        _check_zarr(task, zarr_path, log)
        hdf5_path = _resolve_hdf5(task, args.hdf5)
        log(f"hdf5={hdf5_path}")

        env = RoboCasaEnv(task_name=task, seed=args.seed, image_size=128)
        log(
            f"env={env.env_name} action_dim={env.action_space.shape[0]} "
            f"cams={env.camera_names}"
        )

        n_ok = 0
        n_mismatch = 0
        with h5py.File(hdf5_path, "r") as f:
            demos = _sorted_demo_keys(f["data"])[: args.k]
            log(f"replaying demos: {demos}")
            for ep in demos:
                grp = f[f"data/{ep}"]
                actions = np.asarray(grp["actions"][()], dtype=np.float32)
                states = np.asarray(grp["states"][()])
                initial_state = {
                    "states": states[0],
                    "model": grp.attrs["model_file"],
                    "ep_meta": grp.attrs.get("ep_meta", None),
                }
                if actions.shape[-1] != EXPECTED_ACTION_DIM:
                    raise RuntimeError(f"{ep} action dim {actions.shape[-1]}")
                succ, n_steps, mismatches = replay_demo(env, actions, initial_state)
                if mismatches:
                    n_mismatch += 1
                    for m in mismatches[:5]:
                        log(f"  MISMATCH {ep}: {m}")
                status = "OK" if succ and not mismatches else "FAIL"
                if succ and not mismatches:
                    n_ok += 1
                log(f"  {ep}: {status} success={succ} steps={n_steps} mismatches={len(mismatches)}")

        env.close()
        passed = n_ok == len(demos) and n_mismatch == 0 and len(demos) > 0
        verdict = "PASS" if passed else "FAIL"
        log(
            f"=== {verdict} | {n_ok}/{len(demos)} demos success+parity "
            f"(mismatches_demos={n_mismatch}) ==="
        )
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        verdict = "FAIL"
        log(f"=== {verdict} ===")

    log_path.write_text("\n".join(lines) + "\n")
    log(f"wrote {log_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
