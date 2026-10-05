#!/usr/bin/env python3
"""Plot policy training progress from logs.json — for hackathon slides (fit curve, not final SR claim)."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from oat.common.json_logger import read_json_log


def _sr_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.endswith("mean_success_rate") or c == "mean_success_rate":
            return c
    return None


def summarize(df: pd.DataFrame, task: str, run_dir: pathlib.Path) -> dict:
    sr_col = _sr_column(df)
    out = {
        "task": task,
        "run_dir": str(run_dir),
        "epochs_logged": int(len(df)),
        "final_train_loss": float(df["train_loss"].iloc[-1]) if "train_loss" in df.columns and len(df) else None,
        "final_val_loss": float(df["val_loss"].iloc[-1]) if "val_loss" in df.columns and len(df) else None,
    }
    if sr_col and sr_col in df.columns:
        sr = df[sr_col].dropna()
        if len(sr):
            best_idx = sr.idxmax()
            out["sim_sr_col"] = sr_col
            out["best_sim_sr"] = float(sr.loc[best_idx])
            out["best_sim_sr_epoch"] = int(best_idx)
            out["last_sim_sr"] = float(sr.iloc[-1])
            out["n_rollout_points"] = int(sr.notna().sum())
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        ckpts = sorted(ckpt_dir.glob("ep-*_sr-*.ckpt"))
        out["checkpoints"] = [p.name for p in ckpts[-5:]]
    return out


def plot(df: pd.DataFrame, task: str, out_png: pathlib.Path, summary: dict) -> None:
    sr_col = summary.get("sim_sr_col")
    fig, axes = plt.subplots(1, 2 if sr_col else 1, figsize=(10, 3.5), squeeze=False)
    ax0 = axes[0, 0]

    epochs = range(len(df))
    if "train_loss" in df.columns:
        ax0.plot(epochs, df["train_loss"], label="train loss", color="#2563eb", lw=1.5)
    if "val_loss" in df.columns:
        ax0.plot(epochs, df["val_loss"], label="val loss", color="#94a3b8", lw=1, alpha=0.9)
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("loss")
    ax0.set_title(f"{task.upper()} — policy fit")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)

    if sr_col and sr_col in df.columns:
        ax1 = axes[0, 1]
        sr = df[sr_col]
        mask = sr.notna()
        ax1.plot(list(epochs), sr, "o-", color="#16a34a", lw=1.5, ms=4, label="sim SR (train eval)")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("success rate")
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_title(f"{task.upper()} — sim rollouts during training")
        ax1.grid(True, alpha=0.3)
        if summary.get("best_sim_sr") is not None:
            ep = summary["best_sim_sr_epoch"]
            ax1.axvline(ep, color="#16a34a", ls="--", alpha=0.4, lw=1)
            ax1.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        f"OAT policy from scratch · frozen tokenizer · {summary['epochs_logged']} epochs",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Hydra output dir with logs.json")
    p.add_argument("--task", required=True, choices=["lift", "can", "square"])
    p.add_argument("-o", "--output_dir", default="hackathon_output/progress")
    args = p.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    log_path = run_dir / "logs.json"
    if not log_path.is_file():
        sys.exit(f"missing {log_path}")

    df = read_json_log(str(log_path))
    if df.empty:
        sys.exit(f"empty log: {log_path}")

    out_dir = pathlib.Path(args.output_dir)
    summary = summarize(df, args.task, run_dir)
    plot(df, args.task, out_dir / f"{args.task}_training_curve.png", summary)

    summary_path = out_dir / f"{args.task}_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    shutil.copy2(log_path, out_dir / f"{args.task}_logs.json")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / f'{args.task}_training_curve.png'}")


if __name__ == "__main__":
    main()
