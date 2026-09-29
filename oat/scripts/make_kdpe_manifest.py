#!/usr/bin/env python3
"""Write manifest.json for a KDPE-OAT run (plan sec. 14). Records provenance; no compute."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import click

KDPE_REF_COMMIT = "db0037d09c098626a3da4b3e464322267eccce5f"


def _sha256(path: str, max_bytes: int = 1 << 30) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h, n = hashlib.sha256(), 0
    with p.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            n += len(chunk)
            if n >= max_bytes:
                break
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parent).stdout.strip()


@click.command()
@click.option("--ckpt", required=True)
@click.option("--tokenizer_ckpt", default=None)
@click.option("--R", "r", required=True, type=int, help="canonical executed prefix (n_action_steps) of CS-8")
@click.option("--base_commit", default="8212b7d", show_default=True,
              help="commit the branch was created from (aamas27_selector_baselines)")
@click.option("--bandwidth", default=0.05, show_default=True)
@click.option("--out", default="output/eval/aamas27_selector_baselines/kdpe_n8/manifest.json", show_default=True)
def main(ckpt, tokenizer_ckpt, r, base_commit, bandwidth, out):
    manifest = {
        "method": "kdpe_oat",
        "display_name": "KDPE-style (OAT adaptation)",
        "reference_method": "KDPE",
        "reference_paper": "arXiv:2508.10511",
        "reference_code_commit": KDPE_REF_COMMIT,
        "candidate_generator": "OAT",
        "selection_representation": "last_executed_action",
        "action_format": "libero_delta_xyz_axis_angle_gripper",
        "population_n": 8,
        "action_tokens_k": 8,
        "temperature": 1.0,
        "top_k": 10,
        "execution_horizon_r": r,
        "endpoint_index_0based": r - 1,
        "bandwidth": bandwidth,
        "sigma_pos": bandwidth,
        "sigma_rot": 5 * bandwidth,
        "sigma_grip": 20 * bandwidth,
        "bon_prefix_k": 0,
        "bon_first_temp": 0.0,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_base_commit": base_commit,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "checkpoint_path": ckpt,
        "checkpoint_sha256": _sha256(ckpt),
        "tokenizer_checkpoint_path": tokenizer_ckpt,
        "tokenizer_checkpoint_sha256": _sha256(tokenizer_ckpt) if tokenizer_ckpt else None,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
