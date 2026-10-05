#!/usr/bin/env python3
"""Roll out until N successful episodes; save mp4 + gif for hackathon demo (no SR slide)."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from oat.common.pytorch_util import dict_apply
from oat.env.robomimic.env import RoboMimicEnv
from oat.gymnasium_util.multistep_wrapper import MultiStepWrapper
from oat.gymnasium_util.video_recording_wrapper import VideoRecorder, VideoRecordingWrapper
from oat.policy.base_policy import BasePolicy


def maybe_to_torch(x, device, dtype):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return x


def mp4_to_gif(mp4: pathlib.Path, gif: pathlib.Path, fps: int = 10, width: int = 480) -> None:
    gif.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(mp4),
        "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        str(gif),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _clear_done_flags(env) -> None:
    """Let MultiStepWrapper keep stepping after task success (full demo clip)."""
    msw = env
    if hasattr(msw, "done") and isinstance(msw.done, list) and msw.done:
        msw.done[-1] = False
    inner = msw
    while hasattr(inner, "env"):
        inner = inner.env
        if hasattr(inner, "done"):
            inner.done = False
            break


def run_episode(
    policy,
    task,
    seed,
    out_mp4,
    n_obs,
    n_act,
    max_steps,
    use_k_tokens,
    hold_after_success: int = 80,
    dataset_path: str | None = None,
):
    env = MultiStepWrapper(
        VideoRecordingWrapper(
            RoboMimicEnv(
                task_name=task,
                seed=seed,
                enable_render=True,
                dataset_path=dataset_path,
            ),
            video_recoder=VideoRecorder.create_h264(
                fps=20, codec="h264", input_pix_fmt="rgb24", crf=22,
                thread_type="FRAME", thread_count=1,
            ),
            file_path=str(out_mp4),
            steps_per_render=1,
        ),
        n_obs_steps=n_obs,
        n_action_steps=n_act,
        max_episode_steps=max_steps,
        reward_agg_method="max",
    )
    policy.reset()
    obs, _ = env.reset()
    device, dtype = policy.device, policy.dtype
    success = False
    steps_since_success = 0
    while True:
        obs_dict = dict_apply(
            obs,
            lambda x: maybe_to_torch(x, device, dtype).unsqueeze(0),
        )
        with torch.inference_mode():
            result = policy.predict_action(
                {p: obs_dict[p] for p in policy.get_observation_ports()},
                use_k_tokens=use_k_tokens,
            )
        action = result["action"][0].detach().cpu().numpy()
        obs, reward, done, _, _ = env.step(action)
        chunk_steps = int(np.sum(np.all(np.isfinite(action), axis=-1)))
        if float(reward) >= 1.0:
            if not success:
                success = True
                steps_since_success = 0
        if success:
            steps_since_success += max(chunk_steps, 1)
            _clear_done_flags(env)
            if steps_since_success >= hold_after_success:
                break
        elif done:
            break
        if len(getattr(env, "reward", [])) >= max_steps:
            break
    env.env.video_recoder.stop()
    env.close()
    return success


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("-o", "--output_dir", default="hackathon_output/gifs")
    p.add_argument("--task", required=True, choices=["lift", "can", "square"])
    p.add_argument("--target", type=int, default=1)
    p.add_argument("--max_attempts", type=int, default=80)
    p.add_argument("--seed_base", type=int, default=1000)
    p.add_argument("--use_k_tokens", type=int, default=8)
    p.add_argument("--hold_after_success", type=int, default=80,
                   help="env steps to keep recording after first success")
    p.add_argument("--dataset_path", default=None)
    p.add_argument("-d", "--device", default="cuda:0")
    args = p.parse_args()

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    policy, cfg = BasePolicy.from_checkpoint(args.checkpoint, return_configuration=True)
    policy.to(torch.device(args.device))
    policy.eval()

    n_obs = int(cfg.n_obs_steps)
    n_act = int(cfg.n_action_steps)
    max_steps = int(cfg.task.policy.env_runner.max_episode_steps)

    manifest = []
    n_ok = 0
    for attempt in range(args.max_attempts):
        if n_ok >= args.target:
            break
        seed = args.seed_base + attempt
        stem = f"{args.task}_success_{n_ok:02d}_seed{seed}"
        mp4 = out / f"{stem}.mp4"
        gif = out / f"{stem}.gif"
        print(f"[{args.task}] try {attempt + 1}/{args.max_attempts} seed={seed}", flush=True)
        ok = run_episode(
            policy, args.task, seed, mp4, n_obs, n_act, max_steps, args.use_k_tokens,
            hold_after_success=args.hold_after_success,
            dataset_path=args.dataset_path,
        )
        if not ok or not mp4.is_file() or mp4.stat().st_size < 500:
            mp4.unlink(missing_ok=True)
            print("  miss")
            continue
        mp4_to_gif(mp4, gif)
        manifest.append({"task": args.task, "seed": seed, "mp4": str(mp4), "gif": str(gif)})
        n_ok += 1
        print(f"  saved {gif}")

    (out / f"{args.task}_manifest.json").write_text(json.dumps(manifest, indent=2))
    if n_ok < args.target:
        sys.exit(f"only {n_ok}/{args.target} successes for {args.task}")
    print(f"OK: {n_ok} clip(s) in {out}")


if __name__ == "__main__":
    main()
