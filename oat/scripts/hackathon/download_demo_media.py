#!/usr/bin/env python3
"""Download baseline eval rollout mp4s from aaai27-models for hackathon slides."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--manifest", type=pathlib.Path, required=True)
    p.add_argument("--output_dir", type=pathlib.Path, required=True)
    p.add_argument("--per_task", type=int, default=5)
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(args.repo)

    for task, spec in manifest["tasks"].items():
        prefix = f"eval/matched_s10000/{task}/baseline_n5/media/{task}/"
        mp4s = sorted(f for f in files if f.startswith(prefix) and f.endswith(".mp4"))
        if not mp4s:
            print(f"WARN: no mp4 for {task} under {prefix}", file=sys.stderr)
            continue
        pick = mp4s[: args.per_task]
        out_task = args.output_dir / task
        out_task.mkdir(parents=True, exist_ok=True)
        for hf_path in pick:
            name = pathlib.Path(hf_path).name
            dest = out_task / name
            if dest.is_file() and dest.stat().st_size > 500:
                print(f"  skip {task}/{name}")
                continue
            local = hf_hub_download(args.repo, hf_path, local_dir=str(out_task / "_cache"))
            src = pathlib.Path(local)
            if not src.is_file():
                src = out_task / "_cache" / hf_path
            dest.write_bytes(src.read_bytes())
            print(f"  {task}/{name}")


if __name__ == "__main__":
    main()
