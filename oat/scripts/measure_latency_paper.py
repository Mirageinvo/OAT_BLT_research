#!/usr/bin/env python3
"""Paper Table C / C′ latency — protocol in RESOLUTIONPLAN.md § Latency / Table C.

Measures policy-forward ms (NOT MuJoCo wall-clock) for:
  single  = baseline OAT8 (deployed: predict_action_adaptive; fair_kv: predict_action KV)
  bon{N}  = BoN N∈{8,16,32} vote (same base ckpt)
  awr     = single-sample on Wave2 awr_s10000_<suite>.ckpt (optional; --skip_awr)

Default outs:
  deployed → latency.json
  fair_kv (legacy Single+BoN8+AWR) → latency_fair_kv.json  (LOCKED — do not overwrite lightly)
  fair_kv + (--skip_awr or bon N>8) → latency_fair_kv_n16.json  (does NOT touch locked C′)

Usage (inside docker /workspace/oat):
  python scripts/measure_latency_paper.py --suite can -d cuda:0
  python scripts/measure_latency_paper.py --suite can --fair_kv --skip_awr --bon_ns 16,32
  python scripts/measure_latency_paper.py --suite coffee_press_button --fair_kv --skip_awr \\
      --bon_ns 8,16,32 --base_ckpt my_models/robocasa_coffee_press_button_topk_ep0500_sr0.600.ckpt
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


def _stats(ms: List[float], *, trial_medians: Optional[List[float]] = None) -> Dict[str, Any]:
    arr = np.asarray(ms, dtype=np.float64)
    q25, q75 = np.percentile(arr, [25, 75])
    out: Dict[str, Any] = {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "sem_ms": float(arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0),
        "iqr_ms": float(q75 - q25),
        "p25_ms": float(q25),
        "p75_ms": float(q75),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "reps": int(len(arr)),
        "samples_ms": [float(x) for x in arr.tolist()],
    }
    if trial_medians is not None and len(trial_medians) > 0:
        tm = np.asarray(trial_medians, dtype=np.float64)
        out["n_trials"] = int(len(tm))
        out["trial_medians_ms"] = [float(x) for x in tm.tolist()]
        out["trial_median_mean_ms"] = float(tm.mean())
        out["trial_median_std_ms"] = float(tm.std(ddof=1) if len(tm) > 1 else 0.0)
        # paper-facing: mean±std of per-trial medians
        out["median_ms"] = out["trial_median_mean_ms"]
        out["std_ms"] = out["trial_median_std_ms"]
    return out


def _time_mode_trials(
    fn: Callable[[], None],
    *,
    prep: Callable[[str], None],
    label: str,
    warmup: int,
    reps: int,
    trials: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Independent trials: each = obs-reset + warmup + reps. Paper ± = std of trial medians."""
    all_samples: List[float] = []
    trial_medians: List[float] = []
    for t in range(trials):
        prep(f"{label} trial {t + 1}/{trials}")
        samp = _time_calls(fn, warmup=warmup, reps=reps, device=device)
        trial_medians.append(float(np.median(samp)))
        all_samples.extend(samp)
        print(
            f"    trial {t + 1}/{trials}: median={trial_medians[-1]:.2f} ms "
            f"(n={len(samp)})"
        )
    return _stats(all_samples, trial_medians=trial_medians)


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


def _parse_bon_ns(raw: str) -> List[int]:
    """Parse comma-separated BoN Ns. Empty string → no BoN modes (Single-only runs)."""
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        n = int(tok)
        if n < 1:
            raise click.ClickException(f"invalid bon_ns entry: {tok}")
        out.append(n)
    return out


def _resolve_suite_paths(suite: str) -> tuple[Path, Path]:
    """Return (matched_dir, summary_path_or_empty). Supports robocasa/<task>."""
    matched = ROOT / "output/eval/matched_s10000"
    candidates = [
        matched / suite,
        matched / "robocasa" / suite,
    ]
    for d in candidates:
        for name in ("summary.json", "summary_literal5.json"):
            p = d / name
            if p.is_file():
                return d, p
        if d.is_dir():
            return d, Path()  # dir exists but no summary — need --base_ckpt
    return matched / suite, Path()


@click.command()
@click.option(
    "--suite",
    required=True,
    help="can|…|lift|coffee_press_button|close_drawer|turn_off_sink_faucet|turn_off_microwave",
)
@click.option("-d", "--device", default="cuda:0")
@click.option("--reps", default=10, show_default=True, help="timed reps per trial (protocol: 5–10)")
@click.option(
    "--trials",
    default=1,
    show_default=True,
    help="independent timing trials per mode; paper ± = std of per-trial medians",
)
@click.option("--warmup", default=20, show_default=True)
@click.option("--n_obs", default=8, show_default=True, help="val obs batches to cycle")
@click.option("--seed", default=0, show_default=True, help="timing-loop seed (not Table P env seed)")
@click.option(
    "--out",
    default=None,
    help="explicit out path (overrides default naming)",
)
@click.option(
    "--fair_kv/--deployed",
    default=False,
    show_default=True,
    help=(
        "fair apples-to-apples: Single+AWR use predict_action (KV-cache generate); "
        "BoN unchanged (already generate+KV). Extended runs write latency_fair_kv_n16.json."
    ),
)
@click.option(
    "--skip_awr/--with_awr",
    default=False,
    show_default=True,
    help="skip AWR timing + AWR ckpt/eval requirements (C′ no-AWR extension)",
)
@click.option(
    "--bon_ns",
    default="8",
    show_default=True,
    help="comma-separated BoN N values to time, e.g. 8 or 16,32 or 8,16,32",
)
@click.option(
    "--include_single/--no_single",
    default=True,
    show_default=True,
    help="time Single (KV or deployed). Use --no_single when remasuring only BoN16/32.",
)
@click.option(
    "--base_ckpt",
    default=None,
    help="override base ckpt (required if no summary.json / summary_literal5.json)",
)
@click.option(
    "--awr16_ckpt",
    default=None,
    help=(
        "time AWR16 (BoN16-distill) as modes['awr16'] from this ckpt path. "
        "Does NOT require Wave2 awr_n5 eval. Merges into latency_fair_kv_n16.json."
    ),
)
def main(
    suite: str,
    device: str,
    reps: int,
    trials: int,
    warmup: int,
    n_obs: int,
    seed: int,
    out: Optional[str],
    fair_kv: bool,
    skip_awr: bool,
    bon_ns: str,
    include_single: bool,
    base_ckpt: Optional[str],
    awr16_ckpt: Optional[str],
):
    if reps < 5:
        raise click.ClickException("protocol requires >=5 timed reps")
    if trials < 1:
        raise click.ClickException("--trials must be >=1")

    bon_list = _parse_bon_ns(bon_ns)
    need_base = include_single or bool(bon_list)
    if not need_base and skip_awr and not awr16_ckpt:
        raise click.ClickException(
            "nothing to time: need --include_single and/or --bon_ns and/or AWR/--awr16_ckpt"
        )
    suite_dir, summary_path = _resolve_suite_paths(suite)
    summary: Dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())

    resolved_base: Optional[str] = None
    if need_base:
        if base_ckpt:
            resolved_base = base_ckpt
        elif summary.get("base_ckpt"):
            resolved_base = summary["base_ckpt"]
        else:
            raise click.ClickException(
                f"no base_ckpt: pass --base_ckpt or provide summary under {suite_dir}"
            )
        if not (ROOT / resolved_base).is_file():
            raise click.ClickException(f"missing base ckpt: {resolved_base}")

    awr_ckpt: Optional[str] = None
    if not skip_awr:
        awr_ckpt = summary.get("awr_ckpt") or f"my_models/awr_s10000_{suite}.ckpt"
        if not (ROOT / awr_ckpt).is_file():
            raise click.ClickException(f"missing awr ckpt: {awr_ckpt}")
        awr_eval = suite_dir / "awr_n5" / "eval_log.json"
        if not awr_eval.is_file():
            awr_eval_alt = ROOT / f"output/eval/matched_s10000/{suite}/awr_n5/eval_log.json"
            if not awr_eval_alt.is_file():
                raise click.ClickException(
                    f"Wave2 AWR eval missing ({awr_eval}) — refuse latency without Table P AWR "
                    "(or pass --skip_awr / use --awr16_ckpt)"
                )

    if awr16_ckpt:
        awr16_path = Path(awr16_ckpt)
        if not awr16_path.is_file():
            # also try relative to ROOT
            if (ROOT / awr16_ckpt).is_file():
                awr16_path = ROOT / awr16_ckpt
            else:
                raise click.ClickException(f"missing --awr16_ckpt: {awr16_ckpt}")
        awr16_ckpt = str(awr16_path)

    extended = fair_kv and (
        skip_awr
        or any(n != 8 for n in bon_list)
        or not include_single
        or bool(awr16_ckpt)
    )
    if out:
        out_path = Path(out)
    elif fair_kv and extended:
        out_path = suite_dir / "latency_fair_kv_n16.json"
    elif fair_kv:
        out_path = suite_dir / "latency_fair_kv.json"
    else:
        out_path = suite_dir / "latency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    locked = suite_dir / "latency_fair_kv.json"
    if out_path.resolve() == locked.resolve() and extended:
        raise click.ClickException(
            f"refusing to overwrite locked {locked.name} for extended run; "
            "use default latency_fair_kv_n16.json or pass --out"
        )

    base_ckpt = resolved_base

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
    print(f"suite_dir={suite_dir}")
    print(f"base_ckpt={base_ckpt or '(not loaded)'}")
    print(f"awr_ckpt={awr_ckpt if not skip_awr else '(skipped)'}")
    print(f"awr16_ckpt={awr16_ckpt or '(none)'}")
    print(f"bon_ns={bon_list} include_single={include_single} skip_awr={skip_awr}")
    print(f"device={device} gpu={gpu_name} batch=1 reps={reps} trials={trials} warmup={warmup}")
    print(f"out={out_path}")

    modes: Dict[str, Dict[str, Any]] = {}
    mode_order: List[str] = []
    obs_batches = None
    obs_i = {"i": 0}
    n = 0

    def next_obs():
        o = obs_batches[obs_i["i"] % n]
        obs_i["i"] += 1
        return o

    def _prep_mode(label: str) -> None:
        obs_i["i"] = 0
        if device_t.type == "cuda":
            torch.cuda.synchronize(device_t)
        print(f"Timing {label} (obs reset + {warmup} warmup + {reps} reps)...")

    if need_base:
        print("Loading base policy...")
        base_policy, base_cfg = BasePolicy.from_checkpoint(base_ckpt, return_configuration=True)
        base_policy.to(device_t)
        base_policy.eval()
        obs_batches = _load_obs_batches(base_policy, base_cfg, device_t, n_obs, batch_size=1)
        print(f"Loaded {len(obs_batches)} val obs batches; keys={list(obs_batches[0].keys())}")
        n = len(obs_batches)

        if include_single:
            if fair_kv:

                def run_single():
                    base_policy.predict_action(
                        next_obs(),
                        use_k_tokens=8,
                        temperature=1.0,
                        topk=10,
                    )

                modes["single"] = {
                    "method": "predict_action",
                    "ckpt": base_ckpt,
                    "kwargs": {"use_k_tokens": 8, "temperature": 1.0, "topk": 10},
                    **_time_mode_trials(
                        run_single,
                        prep=_prep_mode,
                        label="single predict_action (KV-cache)",
                        warmup=warmup,
                        reps=reps,
                        trials=trials,
                        device=device_t,
                    ),
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

                modes["single"] = {
                    "method": "predict_action_adaptive",
                    "ckpt": base_ckpt,
                    "kwargs": {
                        "use_k_tokens": 8,
                        "entropy_threshold": 0.0,
                        "temperature": 1.0,
                        "topk": 10,
                    },
                    **_time_mode_trials(
                        run_single,
                        prep=_prep_mode,
                        label="single (OAT8 deployed adaptive)",
                        warmup=warmup,
                        reps=reps,
                        trials=trials,
                        device=device_t,
                    ),
                }
            mode_order.append("single")
            print(
                f"  single = {modes['single']['median_ms']:.2f} ± {modes['single']['std_ms']:.2f} ms "
                f"(trials={modes['single'].get('n_trials', 1)})"
            )

        for n_bon in bon_list:
            mode_key = "bon" if n_bon == 8 else f"bon{n_bon}"

            def run_bon(n_bon=n_bon):
                base_policy.predict_action_adaptive(
                    next_obs(),
                    use_k_tokens=8,
                    entropy_threshold=0.0,
                    temperature=1.0,
                    topk=10,
                    bon_free=n_bon,
                    bon_signal="vote",
                )

            modes[mode_key] = {
                "method": "predict_action_adaptive+bon_free",
                "ckpt": base_ckpt,
                "kwargs": {
                    "use_k_tokens": 8,
                    "entropy_threshold": 0.0,
                    "temperature": 1.0,
                    "topk": 10,
                    "bon_free": n_bon,
                    "bon_signal": "vote",
                },
                **_time_mode_trials(
                    run_bon,
                    prep=_prep_mode,
                    label=f"bon N={n_bon} vote",
                    warmup=warmup,
                    reps=reps,
                    trials=trials,
                    device=device_t,
                ),
            }
            mode_order.append(mode_key)
            print(
                f"  {mode_key} = {modes[mode_key]['median_ms']:.2f} ± {modes[mode_key]['std_ms']:.2f} ms "
                f"(trials={modes[mode_key].get('n_trials', 1)})"
            )

        del base_policy
        if device_t.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("Skipping base policy load (AWR16-only / no single+bon).")

    if not skip_awr:
        assert awr_ckpt is not None
        print("Loading AWR policy (Wave2 ckpt)...")
        awr_policy, awr_cfg = BasePolicy.from_checkpoint(awr_ckpt, return_configuration=True)
        awr_policy.to(device_t)
        awr_policy.eval()
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

            modes["awr"] = {
                "method": "predict_action",
                "ckpt": awr_ckpt,
                "kwargs": {"use_k_tokens": 8, "temperature": 1.0, "topk": 10},
                **_time_mode_trials(
                    run_awr,
                    prep=_prep_mode,
                    label="awr predict_action (KV-cache)",
                    warmup=warmup,
                    reps=reps,
                    trials=trials,
                    device=device_t,
                ),
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

            modes["awr"] = {
                "method": "predict_action_adaptive",
                "ckpt": awr_ckpt,
                "kwargs": {
                    "use_k_tokens": 8,
                    "entropy_threshold": 0.0,
                    "temperature": 1.0,
                    "topk": 10,
                },
                **_time_mode_trials(
                    run_awr,
                    prep=_prep_mode,
                    label="awr (single-sample deployed adaptive)",
                    warmup=warmup,
                    reps=reps,
                    trials=trials,
                    device=device_t,
                ),
            }
        mode_order.append("awr")
        print(
            f"  awr = {modes['awr']['median_ms']:.2f} ± {modes['awr']['std_ms']:.2f} ms "
            f"(trials={modes['awr'].get('n_trials', 1)})"
        )
        del awr_policy
        if device_t.type == "cuda":
            torch.cuda.empty_cache()

    if awr16_ckpt:
        print(f"Loading AWR16 policy ({awr16_ckpt})...")
        awr16_policy, awr16_cfg = BasePolicy.from_checkpoint(awr16_ckpt, return_configuration=True)
        awr16_policy.to(device_t)
        awr16_policy.eval()
        try:
            awr16_obs = _load_obs_batches(awr16_policy, awr16_cfg, device_t, n_obs, batch_size=1)
        except Exception:
            if obs_batches is None:
                raise
            awr16_obs = obs_batches
        n16 = len(awr16_obs)

        def next_obs_awr16():
            o = awr16_obs[obs_i["i"] % n16]
            obs_i["i"] += 1
            return o

        if fair_kv:

            def run_awr16():
                awr16_policy.predict_action(
                    next_obs_awr16(),
                    use_k_tokens=8,
                    temperature=1.0,
                    topk=10,
                )

            modes["awr16"] = {
                "method": "predict_action",
                "ckpt": awr16_ckpt,
                "kwargs": {"use_k_tokens": 8, "temperature": 1.0, "topk": 10},
                "note": "BoN16-distill AWR @100ep (HF Mirageinv/AWR)",
                **_time_mode_trials(
                    run_awr16,
                    prep=_prep_mode,
                    label="awr16 predict_action (KV-cache)",
                    warmup=warmup,
                    reps=reps,
                    trials=trials,
                    device=device_t,
                ),
            }
        else:

            def run_awr16():
                awr16_policy.predict_action_adaptive(
                    next_obs_awr16(),
                    use_k_tokens=8,
                    entropy_threshold=0.0,
                    temperature=1.0,
                    topk=10,
                )

            modes["awr16"] = {
                "method": "predict_action_adaptive",
                "ckpt": awr16_ckpt,
                "kwargs": {
                    "use_k_tokens": 8,
                    "entropy_threshold": 0.0,
                    "temperature": 1.0,
                    "topk": 10,
                },
                "note": "BoN16-distill AWR @100ep (HF Mirageinv/AWR)",
                **_time_mode_trials(
                    run_awr16,
                    prep=_prep_mode,
                    label="awr16 (deployed adaptive)",
                    warmup=warmup,
                    reps=reps,
                    trials=trials,
                    device=device_t,
                ),
            }
        mode_order.append("awr16")
        print(
            f"  awr16 = {modes['awr16']['median_ms']:.2f} ± {modes['awr16']['std_ms']:.2f} ms "
            f"(trials={modes['awr16'].get('n_trials', 1)})"
        )
        del awr16_policy
        if device_t.type == "cuda":
            torch.cuda.empty_cache()

    sr = {
        "baseline": summary.get("baseline_n5") or summary.get("baseline"),
        "bon": summary.get("bon_n8_n5") or summary.get("bon"),
        "awr": summary.get("awr_n5"),
        "note": "SR from Table P / summary — not remeasured",
    }

    git_meta = _git_meta()
    if not git_meta.get("git_commit"):
        raise click.ClickException(
            "git_commit empty — set OAT_GIT_COMMIT (cluster has no .git). "
            "Refuse paper-proof latency without provenance."
        )

    payload = {
        "protocol": (
            "RESOLUTIONPLAN Latency fair-KV extended BoN16/32 no-AWR 2026-07-28"
            if extended
            else (
                "RESOLUTIONPLAN Latency fair-KV 2026-07-18"
                if fair_kv
                else "RESOLUTIONPLAN Latency/Table C paper-proof 2026-07-17"
            )
        ),
        "paper_proof": not fair_kv,
        "fair_kv": fair_kv,
        "fair_single": fair_kv,
        "skip_awr": skip_awr,
        "extended_bon": extended,
        "fairness": {
            "obs_counter_reset_per_mode": True,
            "warmup_per_mode": True,
            "warmup_per_trial": True,
            "cudnn_deterministic": True,
            "batch_size": 1,
            "n_trials": trials,
            "reps_per_trial": reps,
            "mode_order": mode_order,
            "single_path": "predict_action (KV)" if fair_kv else "predict_action_adaptive (no KV)",
            "awr_path": (
                None
                if skip_awr
                else ("predict_action (KV)" if fair_kv else "predict_action_adaptive (no KV)")
            ),
            "bon_path": "predict_action_bon_free / generate (KV)",
            "bon_ns": bon_list,
            "paper_central": "mean of per-trial medians",
            "paper_uncertainty": "std of per-trial medians",
        },
        "suite": suite,
        "suite_dir": str(suite_dir.relative_to(ROOT)),
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
        "trials": trials,
        "reps_per_trial": reps,
        "timing_seed": seed,
        "obs_source": "checkpoint_validation_dataset",
        "n_obs_batches": n_obs,
        "cudnn_deterministic": True,
        **git_meta,
        "base_ckpt": base_ckpt,
        "awr_ckpt": awr_ckpt,
        "awr16_ckpt": awr16_ckpt,
        "base_ckpt_sha256_partial": _sha256(ROOT / base_ckpt) if base_ckpt else None,
        "awr_ckpt_sha256_partial": _sha256(ROOT / awr_ckpt) if awr_ckpt else None,
        "awr16_ckpt_sha256_partial": _sha256(Path(awr16_ckpt)) if awr16_ckpt else None,
        "awr_ckpt_note": (
            "skipped (--skip_awr)"
            if skip_awr
            else "same Wave2 path as matched_s10000/<suite>/awr_n5"
        ),
        "awr16_ckpt_note": (
            "HF Mirageinv/AWR BoN16-distill e100; disk-cycled download"
            if awr16_ckpt
            else None
        ),
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
    # Merge into existing artifact when filling table in phases (Single then BoN).
    if out_path.is_file():
        try:
            prev = json.loads(out_path.read_text())
            prev_modes = prev.get("modes") or {}
            if isinstance(prev_modes, dict) and prev_modes:
                merged = dict(prev_modes)
                merged.update(modes)
                modes = merged
                prev_order = list((prev.get("fairness") or {}).get("mode_order") or [])
                for k in mode_order:
                    if k not in prev_order:
                        prev_order.append(k)
                # keep previous keys that we did not remeasure
                for k in prev_modes:
                    if k not in prev_order:
                        prev_order.append(k)
                mode_order = prev_order
                payload["modes"] = modes
                payload["fairness"]["mode_order"] = mode_order
                payload["merged_from"] = str(out_path)
                print(f"merged modes into existing {out_path} → keys={list(modes)}")
        except Exception as e:
            print(f"WARN: could not merge existing {out_path}: {e}")

    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")
    for k in mode_order:
        if k not in modes:
            continue
        m = modes[k]
        nt = m.get("n_trials", 1)
        print(
            f"  {k}: {m['median_ms']:.2f} ± {m['std_ms']:.2f} ms "
            f"(trial-medians n={nt}; pooled samples={m['reps']})"
        )


if __name__ == "__main__":
    main()
