#!/usr/bin/env python3
"""Run n independent eval experiments; save per-exp SR for dashboard."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("--task", required=True, choices=["lift", "can"])
    p.add_argument("-n", "--num_exp", type=int, default=10)
    p.add_argument("--n_test", type=int, default=10)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--gpu", default="0")
    args = p.parse_args()

    per_exp: list[float] = []
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["MUJOCO_EGL_DEVICE_ID"] = "0"
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env["OAT_USE_UV_RUN"] = "0"

    for i in range(args.num_exp):
        parent = tempfile.mkdtemp(prefix=f"eval_{args.task}_{i}_")
        out_dir = os.path.join(parent, "run")
        cmd = [
            sys.executable,
            str(ROOT / "scripts/eval_policy_sim.py"),
            "--checkpoint",
            args.checkpoint,
            "--output_dir",
            out_dir,
            "--device",
            "cuda:0",
            "--num_exp",
            "1",
            "--n_test",
            str(args.n_test),
            "--test_start_seed",
            str(10000 + i * 1000),
            "--n_parallel_envs",
            "2",
            "--entropy_threshold",
            "0",
            "--use_k_tokens",
            "8",
            "--force",
        ]
        print(f"[{args.task}] exp {i + 1}/{args.num_exp} seed_base={10000 + i * 1000}", flush=True)
        try:
            subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
            log_path = pathlib.Path(out_dir) / "eval_log.json"
            ev = json.loads(log_path.read_text())
            sr = float(ev.get("mean_success_rate_mean", ev.get(f"{args.task}/mean_success_rate_mean", 0.0)))
            per_exp.append(sr)
            print(f"  SR={sr:.3f}", flush=True)
        finally:
            import shutil
            shutil.rmtree(parent, ignore_errors=True)

    out = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "num_exp": args.num_exp,
        "n_test": args.n_test,
        "per_exp_success_rate": per_exp,
        "mean_sr": float(sum(per_exp) / len(per_exp)),
    }
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
