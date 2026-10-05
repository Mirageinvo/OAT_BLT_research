#!/usr/bin/env python3
"""Dark-theme training + eval dashboards for Skoltech POC (HF upload)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from oat.common.json_logger import read_json_log


def read_training_log(path: pathlib.Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return read_json_log(str(path), required_keys=["train_loss", "epoch"])
    return pd.DataFrame(rows)

BG = "#0d1117"
PANEL = "#161b22"
GRID = "#30363d"
CYAN = "#58d9ff"
PINK = "#ff6b9d"
TEAL = "#3ddbd9"
TEXT = "#e6edf3"
MUTED = "#8b949e"


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.alpha": 0.35,
            "font.size": 9,
        }
    )


def _summary_box(ax, lines: list[str]) -> None:
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        color=TEXT,
        bbox=dict(boxstyle="round,pad=0.45", facecolor=PANEL, edgecolor=GRID, alpha=0.95),
    )


def plot_training_dashboard(
    df: pd.DataFrame,
    task: str,
    out_png: pathlib.Path,
    summary: dict,
) -> None:
    _style()
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.32, wspace=0.28)

    ax_main = fig.add_subplot(gs[0, :])
    epochs = np.arange(len(df))
    if "train_loss" in df.columns:
        ax_main.plot(epochs, df["train_loss"], color=CYAN, lw=1.6, label="train loss")
    if "val_loss" in df.columns:
        ax_main.plot(epochs, df["val_loss"], color=MUTED, lw=1.1, alpha=0.85, label="val loss")
    ax_main.set_xlabel("epoch")
    ax_main.set_ylabel("loss")
    ax_main.set_title(f"OAT RoboMimic {task.upper()} — Training Dashboard", fontsize=12, pad=12)
    ax_main.grid(True)
    ax_main.legend(loc="upper right", framealpha=0.2)

    final_train = summary.get("final_train_loss")
    if final_train is not None:
        ax_main.axhline(final_train, color=PINK, ls="--", lw=1, alpha=0.7)
        ax_main.text(
            0.99,
            0.05,
            f"final train={final_train:.3f}",
            transform=ax_main.transAxes,
            ha="right",
            va="bottom",
            color=PINK,
            fontsize=8,
        )

    ax_zoom = fig.add_subplot(gs[1, 0])
    tail = df.tail(min(80, len(df)))
    tail_epochs = np.arange(len(df) - len(tail), len(df))
    if "train_loss" in tail.columns:
        ax_zoom.plot(tail_epochs, tail["train_loss"], color=TEAL, lw=1.8)
    ax_zoom.set_title("Last 80 epochs (zoom)")
    ax_zoom.set_xlabel("epoch")
    ax_zoom.set_ylabel("train loss")
    ax_zoom.grid(True)

    ax_bar = fig.add_subplot(gs[1, 1])
    labels, vals = [], []
    if summary.get("epochs_logged"):
        labels.append("epochs")
        vals.append(summary["epochs_logged"])
    if summary.get("final_train_loss") is not None:
        labels.append("train_loss")
        vals.append(summary["final_train_loss"])
    if summary.get("final_val_loss") is not None:
        labels.append("val_loss")
        vals.append(summary["final_val_loss"])
    if labels:
        colors = [TEAL, CYAN, MUTED][: len(labels)]
        ax_bar.bar(labels, vals, color=colors, edgecolor=GRID)
        ax_bar.set_title("Training summary")
        ax_bar.grid(True, axis="y")

    box_lines = [
        f"task: {task}",
        f"epochs_logged: {summary.get('epochs_logged', '?')}",
        f"final_train_loss: {summary.get('final_train_loss', float('nan')):.4f}"
        if summary.get("final_train_loss") is not None
        else "final_train_loss: n/a",
        f"final_val_loss: {summary.get('final_val_loss', float('nan')):.4f}"
        if summary.get("final_val_loss") is not None
        else "final_val_loss: n/a",
        f"run_dir: {summary.get('run_dir', '')}",
    ]
    fig.text(0.02, 0.98, "\n".join(box_lines), va="top", ha="left", fontsize=8, family="monospace", color=MUTED)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_eval_dashboard(
    per_exp_sr: list[float],
    task: str,
    out_png: pathlib.Path,
    meta: dict | None = None,
) -> dict:
    _style()
    sr = np.asarray(per_exp_sr, dtype=float)
    n = len(sr)
    mean_sr = float(sr.mean())
    std_sr = float(sr.std(ddof=1)) if n > 1 else 0.0
    stderr_sr = std_sr / np.sqrt(n) if n > 1 else 0.0
    best_i = int(sr.argmax())
    best_sr = float(sr[best_i])

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.32, wspace=0.28)

    ax_line = fig.add_subplot(gs[0, :])
    xs = np.arange(1, n + 1)
    ax_line.plot(xs, sr, "o-", color=CYAN, lw=2, ms=7, zorder=3)
    ax_line.scatter([best_i + 1], [best_sr], s=120, facecolors="none", edgecolors=CYAN, lw=2, zorder=4)
    ax_line.axhline(mean_sr, color=PINK, ls="--", lw=1.2, alpha=0.85, label=f"mean={mean_sr:.3f}")
    if n > 1:
        ax_line.fill_between(
            xs,
            mean_sr - stderr_sr,
            mean_sr + stderr_sr,
            color=PINK,
            alpha=0.12,
            label=f"stderr band (+/-{stderr_sr:.3f})",
        )
    ax_line.set_xlim(0.5, n + 0.5)
    ax_line.set_ylim(max(0.0, sr.min() - 0.08), min(1.05, sr.max() + 0.08))
    ax_line.set_xlabel("Experiment index")
    ax_line.set_ylabel("Success rate")
    ax_line.set_title(f"OAT RoboMimic {task.upper()} Evaluation Dashboard", fontsize=12, pad=12)
    ax_line.grid(True)
    ax_line.legend(loc="lower right", framealpha=0.2)
    ax_line.text(
        best_i + 1,
        best_sr,
        f"  best={best_sr:.3f}\n  @exp {best_i + 1}",
        color=CYAN,
        fontsize=8,
        va="bottom",
    )

    ax_box = fig.add_axes([0.02, 0.58, 0.22, 0.28])
    _summary_box(
        ax_box,
        [
            f"num_exp: {n}",
            f"mean_sr: {mean_sr:.3f}",
            f"std_sr: {std_sr:.3f}",
            f"stderr_sr: {stderr_sr:.3f}",
            f"ep_len_mean: {meta.get('ep_len_mean', 'nan') if meta else 'nan'}",
            f"ep_len_std: {meta.get('ep_len_std', 0.0) if meta else 0.0:.2f}",
        ],
    )

    ax_bar = fig.add_subplot(gs[1, 0])
    bar_labels = ["success_rate_mean", "success_rate_std"]
    bar_vals = [mean_sr, std_sr]
    if meta and meta.get("ep_len_mean") is not None and not (isinstance(meta["ep_len_mean"], float) and np.isnan(meta["ep_len_mean"])):
        bar_labels.extend(["episode_len_mean", "episode_len_std"])
        bar_vals.extend([meta["ep_len_mean"], meta.get("ep_len_std", 0.0)])
    colors = [TEAL, PINK] + [MUTED] * max(0, len(bar_labels) - 2)
    ax_bar.bar(bar_labels, bar_vals, color=colors[: len(bar_labels)], edgecolor=GRID)
    ax_bar.set_title("Aggregate eval metrics")
    ax_bar.tick_params(axis="x", rotation=20)
    ax_bar.grid(True, axis="y")

    ax_hist = fig.add_subplot(gs[1, 1])
    bins = max(4, min(8, n))
    ax_hist.hist(sr, bins=bins, color=TEAL, edgecolor=GRID, alpha=0.85)
    ax_hist.axvline(mean_sr, color=PINK, ls="--", lw=1.2, label=f"mean={mean_sr:.3f}")
    ax_hist.axvline(np.median(sr), color=TEXT, ls=":", lw=1, label=f"median={np.median(sr):.3f}")
    ax_hist.set_xlabel("Success rate")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Per-experiment success distribution")
    ax_hist.legend(loc="upper right", fontsize=7, framealpha=0.2)
    ax_hist.grid(True, axis="y")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    summary = {
        "task": task,
        "num_exp": n,
        "per_exp_success_rate": [float(x) for x in sr],
        "mean_sr": mean_sr,
        "std_sr": std_sr,
        "stderr_sr": stderr_sr,
        "best_sr": best_sr,
        "best_exp": best_i + 1,
        **(meta or {}),
    }
    return summary


def summarize_training(df: pd.DataFrame, task: str, run_dir: pathlib.Path) -> dict:
    out = {
        "task": task,
        "run_dir": str(run_dir),
        "epochs_logged": int(len(df)),
        "final_train_loss": float(df["train_loss"].iloc[-1]) if "train_loss" in df.columns and len(df) else None,
        "final_val_loss": float(df["val_loss"].iloc[-1]) if "val_loss" in df.columns and len(df) else None,
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["training", "eval", "all"], default="all")
    p.add_argument("--task", required=True, choices=["lift", "can"])
    p.add_argument("--run_dir", help="Hydra output dir with logs.json (training)")
    p.add_argument("--per_exp_json", help="JSON list of per-experiment SR values (eval)")
    p.add_argument("--eval_log", help="eval_log.json; uses mean_success_rate_mean if single exp")
    p.add_argument("-o", "--output_dir", default="hackathon_output/hf_pack")
    args = p.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    fig_dir = out_dir / "figures"
    meta_dir = out_dir / "eval" / args.task

    if args.mode in ("training", "all"):
        if not args.run_dir:
            sys.exit("--run_dir required for training dashboard")
        run_dir = pathlib.Path(args.run_dir)
        log_path = run_dir / "logs.json"
        df = read_training_log(log_path)
        summary = summarize_training(df, args.task, run_dir)
        plot_training_dashboard(df, args.task, fig_dir / f"{args.task}_training_dashboard.png", summary)
        log_out = out_dir / "logs" / args.task
        log_out.mkdir(parents=True, exist_ok=True)
        (log_out / "training_summary.json").write_text(json.dumps(summary, indent=2))
        import shutil

        shutil.copy2(log_path, log_out / "logs.json")
        print(json.dumps(summary, indent=2))

    if args.mode in ("eval", "all"):
        per_exp: list[float] = []
        meta: dict = {}
        if args.per_exp_json:
            per_exp = json.loads(pathlib.Path(args.per_exp_json).read_text())
            if isinstance(per_exp, dict):
                meta = {k: v for k, v in per_exp.items() if k != "per_exp_success_rate"}
                per_exp = per_exp.get("per_exp_success_rate", [])
        elif args.eval_log:
            ev = json.loads(pathlib.Path(args.eval_log).read_text())
            key = "mean_success_rate_mean"
            if key in ev:
                per_exp = [float(ev[key])]
            meta = {
                "n_test": ev.get("n_test", 10),
                "checkpoint": ev.get("checkpoint"),
                "mean_replans": ev.get("mean_replans_per_episode_mean"),
            }
        else:
            sys.exit("--per_exp_json or --eval_log required for eval dashboard")

        summary = plot_eval_dashboard(per_exp, args.task, fig_dir / f"{args.task}_eval_dashboard.png", meta)
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "eval_dashboard_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
