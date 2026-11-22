"""Per-episode / per-decision CSV logging + manifest for the AAMAS-27 rebuttal runs.

Files written into the eval output_dir (appended once per repeat, never overwritten):
  episodes.csv            one row per episode
  selector_decisions.csv  one row per (episode, replan) when --log_diagnostics is on
  manifest.json           code commit/diff hash, checkpoint sha256, config, scene list hash, GPU
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import time


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(['git', *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


class RebuttalLogger:
    def __init__(self, output_dir: str, variant: str):
        self.output_dir = output_dir
        self.variant = variant
        os.makedirs(output_dir, exist_ok=True)

    def _append(self, name, rows):
        if not rows:
            return
        path = os.path.join(self.output_dir, name)
        new = not os.path.exists(path)
        keys = ['variant'] + [k for k in rows[0].keys()]
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            if new:
                w.writeheader()
            for r in rows:
                w.writerow({'variant': self.variant, **r})

    def append(self, env_runner):
        self._append('episodes.csv', getattr(env_runner, 'last_episode_rows', []))
        self._append('selector_decisions.csv', getattr(env_runner, 'last_decision_rows', []))

    def write_manifest(self, info: dict):
        import torch
        ckpt = info.get('checkpoint')
        scene_list = list(zip(info.get('env_task_names', []), info.get('env_seeds', [])))
        diff = _git('diff', 'HEAD')
        man = dict(
            created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            git_commit=_git('rev-parse', 'HEAD'),
            git_branch=_git('rev-parse', '--abbrev-ref', 'HEAD'),
            git_diff_sha256=(hashlib.sha256(diff.encode()).hexdigest() if diff else None),
            git_dirty_files=_git('status', '--porcelain', '--untracked-files=no'),
            checkpoint_sha256=(_sha256(ckpt) if ckpt and os.path.isfile(ckpt) else None),
            scene_list_sha256=hashlib.sha256(json.dumps(scene_list).encode()).hexdigest(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=(torch.cuda.get_device_name(torch.device(info['device']))
                 if torch.cuda.is_available() and 'cuda' in str(info.get('device')) else None),
            host=platform.node(),
            mujoco_gl=os.environ.get('MUJOCO_GL'),
            **{k: v for k, v in info.items()},
        )
        with open(os.path.join(self.output_dir, 'manifest.json'), 'w') as f:
            json.dump(man, f, indent=2, default=str)
