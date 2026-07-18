#!/usr/bin/env python3
"""Paper Table C latency — protocol in RESOLUTIONPLAN.md § Latency / Table C.

Measures policy-forward ms (NOT MuJoCo wall-clock) for:
  single  = baseline OAT8 (predict_action_adaptive, k=8, entropy_threshold=0)
  bon     = BoN N=8 vote (same base ckpt)
  awr     = single-sample on Wave2 awr_s10000_<suite>.ckpt

Writes output/eval/matched_s10000/<suite>/latency.json with git/host/ckpt metadata.

Usage (inside docker /workspace/oat):
  python scripts/measure_latency_paper.py --suite can -d cuda:0
  python scripts/measure_latency_paper.py --suite box-close --reps 10 --warmup 20
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import click
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

from oat.policy.base_policy import BasePolicy


def _git_meta() -> Dict[str, Any]:
    """Resolve git identity: env override (cluster has no .git) → walk parents → empty."""
    env_commit = os.environ.get("OAT_GIT_COMMIT", "").strip()
    env_branch = os.environ.get("OAT_GIT_BRANCH", "").strip()
    env_dirty = os.environ.get("OAT_GIT_DIRTY", "").strip()
    if env_commit:
        dirty = env_dirty.lower() in ("1", "true", "yes", "dirty") if env_dirty else False
        return {
            "git_commit": env_commit,
            "git_branch": env_branch or "unknown",
            "git_dirty": dirty,
            "git_source": "env:OAT_GIT_*",
        }

    def run(args: List[str], cwd: Path) -> str:
        try:
            return subprocess.check_output(args, cwd=str(cwd), stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""

    # walk ROOT and parents for .git (host checkout); also OAT_GIT_DIR
    candidates = []
    if os.environ.get("OAT_GIT_DIR"):
        candidates.append(Path(os.environ["OAT_GIT_DIR"]))
    candidates.append(ROOT)
    candidates.extend(ROOT.parents)
    for cwd in candidates:
        if not (cwd / ".git").exists() and cwd != Path(os.environ.get("OAT_GIT_DIR", "")):
            continue
        commit = run(["git", "rev-parse", "HEAD"], cwd)
        if not commit:
            continue
        branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
        dirty = bool(run(["git", "status", "--porcelain"], cwd))
        return {
            "git_commit": commit,
            "git_branch": branch,
            "git_dirty": dirty,
            "git_source": f"git:{cwd}",
        }
    return {
        "git_commit": "",
        "git_branch": "",
        "git_dirty": False,
        "git_source": "unavailable",
    }


def _sha256(path: Path, max_bytes: int = 64 * 1024 * 1024) -> Optional[str]:
    """Hash file; for huge ckpts hash first+last chunk + size (fast identity check)."""
    if not path.is_file():
        return None
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(f"size={size}\n".encode())
    with open(path, "rb") as f:
        h.update(f.read(min(max_bytes, size)))
        if size > max_bytes:
            f.seek(max(0, size - max_bytes))
            h.update(f.read(max_bytes))
    return h.hexdigest()


def _stats(ms: List[float]) -> Dict[str, float]:
    arr = np.asarray(ms, dtype=np.float64)
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "iqr_ms": float(q75 - q25),
        "p25_ms": float(q25),
        "p75_ms": float(q75),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "reps": int(len(arr)),
        "samples_ms": [float(x) for x in arr.tolist()],
    }


def _load_obs_batches(policy, cfg, device, n_batches: int, batch_size: int = 1):
    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    val_dataset = dataset.get_validation_dataset()
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    ports = policy.get_observation_ports()
    batches = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batches.append({k: batch["obs"][k].to(device) for k in ports})
    if not batches:
        raise RuntimeError("No val obs batches loaded — empty validation dataset?")
    return batches


def _time_calls(
    fn: Callable[[], None],
    *,
    warmup: int,
    reps: int,
    device: torch.device,
) -> List[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(reps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


@click.command()
@click.option("--suite", required=True, help="can|coffee-pull|stick-pull|disassemble|box-close|square")
@click.option("-d", "--device", default="cuda:0")
@click.option("--reps", default=10, show_default=True, help="timed reps per mode (protocol: 5–10)")
@click.option("--warmup", default=20, show_default=True)
@click.option("--n_obs", default=8, show_default=True, help="val obs batches to cycle")
@click.option("--seed", default=0, show_default=True, help="timing-loop seed (not Table P env seed)")
@click.option(
    "--out",
    default=None,
    help="default: matched_s10000/<suite>/latency.json (or latency_fair_kv.json with --fair_kv)",
)
@click.option(
    "--fair_kv/--deployed",
    default=False,
    show_default=True,
    help=(
        "fair apples-to-apples: Single+AWR use predict_action (KV-cache generate); "
        "BoN unchanged (already generate+KV). Writes latency_fair_kv.json; does NOT "
        "overwrite paper Table C latency.json."
    ),
)
def main(
    suite: str,
    device: str,
    reps: int,
    warmup: int,
    n_obs: int,
    seed: int,
    out: Optional[str],
    fair_kv: bool,
):
    if reps < 5:
        raise click.ClickException("protocol requires >=5 timed reps")

    summary_path = ROOT / f"output/eval/matched_s10000/{suite}/summary.json"
    if not summary_path.is_file():
        raise click.ClickException(f"missing Wave summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    base_ckpt = summary["base_ckpt"]
    awr_ckpt = summary.get("awr_ckpt") or f"my_models/awr_s10000_{suite}.ckpt"
    for name, p in [("base", base_ckpt), ("awr", awr_ckpt)]:
        if not (ROOT / p).is_file():
            raise click.ClickException(f"missing {name} ckpt: {p}")
    awr_eval = ROOT / f"output/eval/matched_s10000/{suite}/awr_n5/eval_log.json"
    if not awr_eval.is_file():
        raise click.ClickException(f"Wave2 AWR eval missing ({awr_eval}) — refuse latency without Table P AWR")

    if out:
        out_path = Path(out)
    elif fair_kv:
        out_path = ROOT / f"output/eval/matched_s10000/{suite}/latency_fair_kv.json"
    else:
        out_path = ROOT / f"output/eval/matched_s10000/{suite}/latency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic timing stack (protocol §2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device_t = torch.device(device)
    gpu_name = ""
    cuda_version = ""
    if device_t.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device_t)
        cuda_version = getattr(torch.version, "cuda", "") or ""

    mode_tag = "FAIR_KV" if fair_kv else "DEPLOYED"
    print(f"=== PAPER LATENCY {suite} [{mode_tag}] ===")
    print(f"base_ckpt={base_ckpt}")
    print(f"awr_ckpt={awr_ckpt}")
    print(f"device={device} gpu={gpu_name} reps={reps} warmup={warmup}")

    # --- base policy: single + bon ---
    print("Loading base policy...")
    base_policy, base_cfg = BasePolicy.from_checkpoint(base_ckpt, return_configuration=True)
    base_policy.to(device_t)
    base_policy.eval()
    obs_batches = _load_obs_batches(base_policy, base_cfg, device_t, n_obs, batch_size=1)
    print(f"Loaded {len(obs_batches)} val obs batches; keys={list(obs_batches[0].keys())}")
    n = len(obs_batches)
    obs_i = {"i": 0}

    def next_obs():
        o = obs_batches[obs_i["i"] % n]
        obs_i["i"] += 1
        return o

    modes: Dict[str, Dict[str, Any]] = {}

    def _prep_mode(label: str) -> None:
        """Fair mode start: same obs sequence, fresh CUDA sync (no order-effect carry)."""
        obs_i["i"] = 0
        if device_t.type == "cuda":
            torch.cuda.synchronize(device_t)
        print(f"Timing {label} (obs reset + {warmup} warmup + {reps} reps)...")

    if fair_kv:
        def run_single():
            base_policy.predict_action(
                next_obs(),
                use_k_tokens=8,
                temperature=1.0,
                topk=10,
            )

        _prep_mode("single predict_action (KV-cache)")
        modes["single"] = {
            "method": "predict_action",
            "ckpt": base_ckpt,
            "kwargs": {"use_k_tokens": 8, "temperature": 1.0, "topk": 10},
            **_stats(_time_calls(run_single, warmup=warmup, reps=reps, device=device_t)),
        }
    else:
        def run_single():
            base_policy.predict_action_adaptive(
                next_obs(),
                use_k_tokens=8,
                entropy_threshold=0.0,
                temperature=1.0,
                topk=10,
            )

        _prep_mode("single (OAT8 deployed adaptive)")
        modes["single"] = {
            "method": "predict_action_adaptive",
            "ckpt": base_ckpt,
            "kwargs": {"use_k_tokens": 8, "entropy_threshold": 0.0, "temperature": 1.0, "topk": 10},
            **_stats(_time_calls(run_single, warmup=warmup, reps=reps, device=device_t)),
        }
    print(f"  single median={modes['single']['median_ms']:.2f} ms")

    def run_bon():
        base_policy.predict_action_adaptive(
            next_obs(),
            use_k_tokens=8,
            entropy_threshold=0.0,
            temperature=1.0,
            topk=10,
            bon_free=8,
            bon_signal="vote",
        )

    _prep_mode("bon N=8 vote")
    modes["bon"] = {
        "method": "predict_action_adaptive+bon_free",
        "ckpt": base_ckpt,
        "kwargs": {
            "use_k_tokens": 8,
            "entropy_threshold": 0.0,
            "temperature": 1.0,
            "topk": 10,
            "bon_free": 8,
            "bon_signal": "vote",
        },
        **_stats(_time_calls(run_bon, warmup=warmup, reps=reps, device=device_t)),
    }
    print(f"  bon median={modes['bon']['median_ms']:.2f} ms")

    # free base VRAM before AWR
    del base_policy
    if device_t.type == "cuda":
        torch.cuda.empty_cache()

    print("Loading AWR policy (Wave2 ckpt)...")
    awr_policy, awr_cfg = BasePolicy.from_checkpoint(awr_ckpt, return_configuration=True)
    awr_policy.to(device_t)
    awr_policy.eval()
    # AWR trained on same task — reuse obs from base val if configs match; else reload
    try:
        awr_obs = _load_obs_batches(awr_policy, awr_cfg, device_t, n_obs, batch_size=1)
    except Exception:
        awr_obs = obs_batches
    n_a = len(awr_obs)

    def next_obs_awr():
        o = awr_obs[obs_i["i"] % n_a]
        obs_i["i"] += 1
        return o

    if fair_kv:
        def run_awr():
            awr_policy.predict_action(
                next_obs_awr(),
                use_k_tokens=8,
                temperature=1.0,
                topk=10,
            )

        _prep_mode("awr predict_action (KV-cache)")
        modes["awr"] = {
            "method": "predict_action",
            "ckpt": awr_ckpt,
            "kwargs": {"use_k_tokens": 8, "temperature": 1.0, "topk": 10},
            **_stats(_time_calls(run_awr, warmup=warmup, reps=reps, device=device_t)),
        }
    else:
        def run_awr():
            awr_policy.predict_action_adaptive(
                next_obs_awr(),
                use_k_tokens=8,
                entropy_threshold=0.0,
                temperature=1.0,
                topk=10,
            )

        _prep_mode("awr (single-sample deployed adaptive)")
        modes["awr"] = {
            "method": "predict_action_adaptive",
            "ckpt": awr_ckpt,
            "kwargs": {"use_k_tokens": 8, "entropy_threshold": 0.0, "temperature": 1.0, "topk": 10},
            **_stats(_time_calls(run_awr, warmup=warmup, reps=reps, device=device_t)),
        }
    print(f"  awr median={modes['awr']['median_ms']:.2f} ms")

    sr = {
        "baseline": summary.get("baseline_n5"),
        "bon": summary.get("bon_n8_n5"),
        "awr": summary.get("awr_n5"),
        "note": "SR from Table P / summary.json — not remeasured",
    }

    git_meta = _git_meta()
    if not git_meta.get("git_commit"):
        raise click.ClickException(
            "git_commit empty — set OAT_GIT_COMMIT (cluster has no .git). "
            "Refuse paper-proof latency without provenance."
        )

    payload = {
        "protocol": (
            "RESOLUTIONPLAN Latency fair-KV 2026-07-18"
            if fair_kv
            else "RESOLUTIONPLAN Latency/Table C paper-proof 2026-07-17"
        ),
        "paper_proof": not fair_kv,
        "fair_kv": fair_kv,
        "fair_single": fair_kv,  # alias for rebuttal naming
        "fairness": {
            "obs_counter_reset_per_mode": True,
            "warmup_per_mode": True,
            "cudnn_deterministic": True,
            "mode_order": ["single", "bon", "awr"],
            "single_path": "predict_action (KV)" if fair_kv else "predict_action_adaptive (no KV)",
            "awr_path": "predict_action (KV)" if fair_kv else "predict_action_adaptive (no KV)",
            "bon_path": "predict_action_bon_free / generate (KV)",
        },
        "suite": suite,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "torch_version": torch.__version__,
        "device": device,
        "docker_image": os.environ.get("OAT_DOCKER_IMAGE", os.environ.get("HOSTNAME", "")),
        "batch_size": 1,
        "warmup_excluded": True,
        "warmup": warmup,
        "timing_seed": seed,
        "obs_source": "checkpoint_validation_dataset",
        "n_obs_batches": n_obs,
        "cudnn_deterministic": True,
        **git_meta,
        "base_ckpt": base_ckpt,
        "awr_ckpt": awr_ckpt,
        "base_ckpt_sha256_partial": _sha256(ROOT / base_ckpt),
        "awr_ckpt_sha256_partial": _sha256(ROOT / awr_ckpt),
        "awr_ckpt_note": "same Wave2 path as matched_s10000/<suite>/awr_n5",
        "table_p_sr": sr,
        "modes": modes,
        "cost_note": (
            "ms = policy-forward only (vision+AR[+BoN select]); "
            "episode wall-clock dominated by MuJoCo/render — not included. "
            + (
                "FAIR_KV: Single/AWR use predict_action KV-cache (apples-to-apples with BoN generate)."
                if fair_kv
                else "DEPLOYED: Single/AWR use predict_action_adaptive (Table C main)."
            )
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")
    for k in ("single", "bon", "awr"):
        m = modes[k]
        print(f"  {k}: median={m['median_ms']:.2f}  mean±std={m['mean_ms']:.2f}±{m['std_ms']:.2f}  n={m['reps']}")


if __name__ == "__main__":
    main()
