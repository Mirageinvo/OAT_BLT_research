#!/usr/bin/env python3
"""Check local baseline asset tree against baseline_manifest.json."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def ok(p: pathlib.Path) -> bool:
    return p.is_file() and p.stat().st_size > 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("scripts/hackathon/baseline_manifest.json"))
    p.add_argument("--stage", type=pathlib.Path, default=pathlib.Path("hackathon_assets"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("hackathon_output"))
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    missing = []
    print("RoboMimic baseline asset check\n")
    print(f"{'Task':<8} {'Policy':<6} {'Tok':<6} {'Summary':<8} {'Demo mp4':<10} Paper SR")
    print("-" * 72)

    for task, spec in manifest["tasks"].items():
        pol = args.stage / "policies" / f"{task}.ckpt"
        tok = args.stage / "tokenizers" / f"{task}.ckpt"
        summ = args.out / spec["eval_summary_hf"]
        media_dir = args.out / "demo" / "media" / task
        n_mp4 = len(list(media_dir.glob("*.mp4"))) if media_dir.is_dir() else 0

        flags = [
            "OK" if ok(pol) else "MISS",
            "OK" if ok(tok) else "MISS",
            "OK" if summ.is_file() else "MISS",
            f"{n_mp4}" if n_mp4 else "MISS",
        ]
        for label, path in [("policy", pol), ("tokenizer", tok), ("summary", summ)]:
            if not (path.is_file() and path.stat().st_size > 100):
                missing.append(str(path))

        print(f"{task:<8} {flags[0]:<6} {flags[1]:<6} {flags[2]:<8} {flags[3]:<10} {spec['baseline_sr_pct']}")

    print("\nHF repos:")
    print(f"  weights:  {manifest['model_repo']}")
    print(f"  datasets: {manifest['dataset_repo']}")
    print(f"  eval:     {manifest['eval_repo']}")

    if missing:
        print("\nMissing:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)
    print("\nAll required baseline assets present.")


if __name__ == "__main__":
    main()
