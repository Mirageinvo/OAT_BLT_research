#!/usr/bin/env python3
"""One slide: train loss for lift/can/square side by side."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from oat.common.json_logger import read_json_log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--progress_dir", default="hackathon_output/progress")
    p.add_argument("-o", "--output", default="hackathon_output/progress/all_tasks_training.png")
    args = p.parse_args()

    prog = pathlib.Path(args.progress_dir)
    tasks = ["lift", "can", "square"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=False)

    for ax, task in zip(axes, tasks):
        log_copy = prog / f"{task}_logs.json"
        summary_path = prog / f"{task}_training_summary.json"
        if not log_copy.is_file():
            ax.set_title(f"{task} (no log)")
            ax.axis("off")
            continue
        df = read_json_log(str(log_copy))
        ax.plot(df["train_loss"], color="#2563eb", lw=1.5)
        ax.set_title(task.upper())
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
        ax.grid(True, alpha=0.3)
        if summary_path.is_file():
            s = json.loads(summary_path.read_text())
            ax.text(
                0.03, 0.97,
                f"{s.get('epochs_logged', '?')} ep\nbest sim {s.get('best_sim_sr', 0):.0%}" if s.get("best_sim_sr") is not None else f"{s.get('epochs_logged', '?')} ep",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    fig.suptitle("Policy training from scratch (frozen OAT tokenizer)", fontsize=11)
    fig.tight_layout()
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
