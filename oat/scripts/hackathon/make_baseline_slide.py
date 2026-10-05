#!/usr/bin/env python3
"""Build training / baseline summary cards for slides (no logs.json required on HF)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import yaml


def load_yaml(p: pathlib.Path) -> dict:
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("scripts/hackathon/baseline_manifest.json"))
    p.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("hackathon_output/progress"))
    p.add_argument("--eval_root", type=pathlib.Path, default=pathlib.Path("hackathon_output"))
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (task, spec) in zip(axes, manifest["tasks"].items()):
        summary_path = args.eval_root / spec["eval_summary_hf"]
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        hydra_path = args.eval_root / spec["hydra_config_hf"]
        cfg = load_yaml(hydra_path)
        train_cfg = cfg.get("training", {})
        num_epochs = train_cfg.get("num_epochs")
        num_demo = train_cfg.get("num_demo") or spec.get("dataset_episodes", 200)

        pol_name = pathlib.Path(spec["policy_hf"]).name
        tok_name = pathlib.Path(spec["tokenizer_hf"]).name
        baseline = summary.get("baseline_n5", {})
        sr_pct = baseline.get("pct") or spec["baseline_sr_pct"]
        sr_mean = baseline.get("mean") or spec["baseline_sr_mean"]

        row = {
            "task": task,
            "title": spec["title_ru"],
            "tokenizer": spec["tokenizer_hf"],
            "policy": spec["policy_hf"],
            "policy_ckpt_epoch": spec["policy_epoch"],
            "dataset": spec["dataset"],
            "num_demo": num_demo,
            "train_num_epochs": num_epochs,
            "paper_baseline_sr_pct": sr_pct,
            "paper_baseline_sr_mean": sr_mean,
            "eval_protocol": manifest["protocol"],
        }
        rows.append(row)
        (args.out_dir / f"{task}_baseline_summary.json").write_text(json.dumps(row, indent=2))

        text = (
            f"{spec['title_ru']}\n\n"
            f"Dataset: {spec['dataset']} ({num_demo} demos)\n"
            f"Tokenizer: {tok_name}\n"
            f"Policy: {pol_name}\n"
            f"Train budget: {num_epochs or '?'} epochs\n"
            f"Paper baseline SR: {sr_pct}\n"
            f"(matched_s10000, n=5×50, seed≥10000)"
        )
        ax.axis("off")
        ax.text(0.05, 0.95, text, va="top", ha="left", fontsize=9, family="monospace",
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="#f8fafc", edgecolor="#cbd5e1"))
        ax.set_title(task.upper(), fontweight="bold")

    fig.suptitle("OAT RoboMimic baseline (HF paper ckpts)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_png = args.out_dir / "baseline_training_cards.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    combined = {
        "repos": {
            "weights": manifest["model_repo"],
            "datasets": manifest["dataset_repo"],
            "eval": manifest["eval_repo"],
        },
        "tasks": rows,
    }
    (args.out_dir / "baseline_all_summary.json").write_text(json.dumps(combined, indent=2))
    print(f"Wrote {out_png}")
    print(f"Wrote {args.out_dir}/baseline_all_summary.json")


if __name__ == "__main__":
    main()
