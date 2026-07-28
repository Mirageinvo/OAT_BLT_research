#!/usr/bin/env bash
# Wait until RoboCasa turn_off_microwave Wave1 BoN (all 5 seeds) finishes,
# then: aggregate → patch RESULTS.md → upload README + eval artifacts to
# https://huggingface.co/hackhackhack66666/last
#
# Run on CLUSTER HOST (not inside docker) so HF token works:
#   nohup bash scripts/watch_mw_wave1_then_hf_last.sh >> logs/watch_mw_hf_last.log 2>&1 &
#
# Prerequisites:
#   - hf auth whoami  → hackhackhack66666  (token in ~/.cache/huggingface/token)
#   - Wave1 microwave still running or already finished
set -euo pipefail

# Resolve oat root: host mount or docker workdir
if [[ -d /home/askhabaliev_gs/mipt_paper/oat ]]; then
  ROOT=/home/askhabaliev_gs/mipt_paper/oat
elif [[ -d /workspace/oat ]]; then
  ROOT=/workspace/oat
else
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "${ROOT}"
mkdir -p logs

LOG="${LOG:-logs/watch_mw_hf_last.log}"
HF_REPO="${HF_REPO:-hackhackhack66666/last}"
HF_BIN="${HF_BIN:-}"
TASK=turn_off_microwave
EVAL_ROOT="output/eval/matched_s10000/robocasa/${TASK}"
SEEDS=(10000 10001 10002 10003 10004)
POLL_SEC="${POLL_SEC:-60}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "${LOG}"; }

resolve_hf() {
  if [[ -n "${HF_BIN}" && -x "${HF_BIN}" ]]; then return 0; fi
  for c in /home/askhabaliev_gs/.local/bin/hf "$(command -v hf || true)"; do
    if [[ -n "${c}" && -x "${c}" ]]; then HF_BIN="${c}"; return 0; fi
  done
  log "ERROR: hf CLI not found"; exit 1
}

check_hf_auth() {
  resolve_hf
  local who
  who="$("${HF_BIN}" auth whoami 2>&1)" || true
  if ! echo "${who}" | grep -q 'hackhackhack66666\|Logged in\|user:'; then
    log "ERROR: HF not logged in. On host: hf auth login  (need write access to ${HF_REPO})"
    log "whoami output: ${who}"
    exit 1
  fi
  log "HF auth OK: $(echo "${who}" | tr '\n' ' ')"
}

all_bon_done() {
  local s
  for s in "${SEEDS[@]}"; do
    [[ -f "${EVAL_ROOT}/bon_n8_seed${s}/eval_log.json" ]] || return 1
  done
  return 0
}

baseline_done() {
  local s
  for s in "${SEEDS[@]}"; do
    [[ -f "${EVAL_ROOT}/baseline_seed${s}/eval_log.json" ]] || return 1
  done
  return 0
}

count_bon() {
  local n=0 s
  for s in "${SEEDS[@]}"; do
    [[ -f "${EVAL_ROOT}/bon_n8_seed${s}/eval_log.json" ]] && n=$((n + 1))
  done
  echo "${n}"
}

# --- wait ---
check_hf_auth
log "=== watch microwave Wave1 → HF ${HF_REPO} ==="
log "waiting for ${EVAL_ROOT}/bon_n8_seed{10000..10004}/eval_log.json (poll=${POLL_SEC}s)"

if ! baseline_done; then
  log "WARN: baseline not complete yet — still waiting for full Wave1"
fi

while ! all_bon_done; do
  log "progress BonN8 $(count_bon)/5"
  sleep "${POLL_SEC}"
done
log "ALL BoN seeds present — packaging"

# --- aggregate ---
PY=""
for c in \
  "${ROOT}/.venv_robocasa/bin/python" \
  "${ROOT}/.venv/bin/python" \
  /workspace/oat/_runtime_uv_python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10 \
  "$(command -v python3)"; do
  if [[ -n "${c}" && -x "${c}" ]]; then PY="${c}"; break; fi
done
[[ -n "${PY}" ]] || { log "ERROR: no python"; exit 1; }

"${PY}" scripts/aggregate_robocasa_literal5.py --root "${EVAL_ROOT}" | tee -a "${LOG}"
[[ -f "${EVAL_ROOT}/summary_literal5.json" ]] || { log "ERROR: missing summary"; exit 1; }

# --- build staging + README + patch RESULTS ---
STAGE="$(mktemp -d /tmp/hf_last_XXXXXX)"
trap 'rm -rf "${STAGE}"' EXIT

"${PY}" - "${ROOT}" "${EVAL_ROOT}" "${STAGE}" <<'PY'
import json, math, pathlib, re, sys, datetime

root = pathlib.Path(sys.argv[1])
mw_root = pathlib.Path(sys.argv[2])
stage = pathlib.Path(sys.argv[3])
sink_root = root / "output/eval/matched_s10000/robocasa/turn_off_sink_faucet"

def load_sum(p: pathlib.Path):
    return json.loads(p.read_text()) if p.is_file() else None

def fmt_pct(x, digits=1):
    return f"{100*x:.{digits}f}"

def method_line(d, key):
    m = d["methods"][key]
    seeds = {k: round(100 * v, 1) for k, v in m["per_seed_sr"].items()}
    return (
        f"{fmt_pct(m['mean'])}±{fmt_pct(m['sem'])}%",
        seeds,
        m,
    )

def delta_line(d):
    dd = d["deltas"]["delta_bon_n8_minus_baseline"]
    return f"{100*dd['delta']:+.1f}±{100*dd['sem_delta']:.1f} pp"

rows = []
for name, p in [
    ("turn_off_sink_faucet", sink_root / "summary_literal5.json"),
    ("turn_off_microwave", mw_root / "summary_literal5.json"),
]:
    d = load_sum(p)
    if not d or "baseline" not in d.get("methods", {}):
        continue
    b, bs, _ = method_line(d, "baseline")
    if "bon_n8" in d["methods"]:
        n8, ns, _ = method_line(d, "bon_n8")
        delta = delta_line(d)
    else:
        n8, ns, delta = "—", {}, "—"
    rows.append((name, b, n8, delta, bs, ns, d))

# copy artifacts
for task, r in [("turn_off_sink_faucet", sink_root), ("turn_off_microwave", mw_root)]:
    if not r.is_dir():
        continue
    dest = stage / "robocasa" / task
    dest.mkdir(parents=True, exist_ok=True)
    for p in r.rglob("eval_log.json"):
        rel = p.relative_to(r)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(p.read_bytes())
    for name in ("summary_literal5.json", "wave1_literal5.log"):
        src = r / name
        if src.is_file():
            (dest / name).write_bytes(src.read_bytes())

# README (with HF model-card YAML to avoid empty-metadata warning)
lines = []
lines.append("---\n")
lines.append("license: mit\n")
lines.append("tags:\n")
lines.append("- robotics\n")
lines.append("- robocasa\n")
lines.append("- oat\n")
lines.append("- evaluation\n")
lines.append("---\n\n")
lines.append("# RoboCasa Wave1 — last tasks (literal-5)\n\n")
lines.append(f"Uploaded: `{datetime.datetime.now(datetime.timezone.utc).isoformat()}`\n\n")
lines.append("Protocol: `test_start_seed ∈ {10000..10004}`, `-n 1 --n_test 50`, OAT8, BoN `--bon_free 8 --bon_signal vote`.\n\n")
lines.append("Aggregation: mean±SEM over 5 seeds; `Δ = mean(BoN)−mean(base)`, `SEM_Δ = sqrt(SEM_b²+SEM_m²)` (unpaired).\n\n")
lines.append("## Table P (Wave1)\n\n")
lines.append("| task | baseline mean±SEM | BoN N=8 | Δ_BoN±SEM_Δ |\n")
lines.append("|------|-------------------|---------|-------------|\n")
for name, b, n8, delta, *_ in rows:
    lines.append(f"| `{name}` | **{b}** | **{n8}** | **{delta}** |\n")
lines.append("\n## Per-seed SR (%)\n")
for name, b, n8, delta, bs, ns, d in rows:
    lines.append(f"\n### `{name}`\n\n")
    lines.append(f"- baseline seeds: `{bs}`\n")
    lines.append(f"- BoN8 seeds: `{ns}`\n")
    lines.append(f"- Δ_BoN: **{delta}**\n")
lines.append("\n## Artifacts in this repo\n\n")
lines.append("```text\n")
lines.append("robocasa/<task>/\n")
lines.append("  summary_literal5.json\n")
lines.append("  baseline_seed{10000..10004}/eval_log.json\n")
lines.append("  bon_n8_seed{10000..10004}/eval_log.json\n")
lines.append("  wave1_literal5.log   # if present\n")
lines.append("```\n\n")
lines.append("**Not included** (too large / elsewhere): policy TopK ckpts (~0.9GB). See `hackhackhack66666/rc-model-two-last`.\n\n")
lines.append("## Reproduce aggregate\n\n")
lines.append("```bash\n")
lines.append("python scripts/aggregate_robocasa_literal5.py \\\n")
lines.append("  --root output/eval/matched_s10000/robocasa/turn_off_microwave\n")
lines.append("```\n")
(stage / "README.md").write_text("".join(lines), encoding="utf-8")
print("README written")
for name, b, n8, delta, *_ in rows:
    print(f"ROW {name} | {b} | {n8} | {delta}")

# patch RESULTS.md microwave (+ ensure sink DONE)
res = root / "RESULTS.md"
if res.is_file() and any(r[0] == "turn_off_microwave" for r in rows):
    text = res.read_text(encoding="utf-8")
    mw = next(r for r in rows if r[0] == "turn_off_microwave")
    _, b, n8, delta, *_ = mw
    # main Table P row
    text2, n = re.subn(
        r"\| turn_off_microwave \| \*\*ep-0500 @0\.620\*\* lock \|.*?\| `my_models/robocasa_turn_off_microwave[^\|]*\|\n",
        f"| turn_off_microwave | **ep-0500 @0.620** lock | **{b}** | **{n8}** | — | **{delta}** | — | Wave1 **DONE** · `summary_literal5.json` · `my_models/robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt` |\n",
        text,
        count=1,
    )
    # RoboCasa detail table
    text2, n2 = re.subn(
        r"\| turn_off_microwave \| \*\*`my_models/robocasa_turn_off_microwave_topk_ep0500_sr0\.620\.ckpt`\*\* \|.*?\| `matched_s10000/robocasa/turn_off_microwave/`[^\|]*\|\n",
        f"| turn_off_microwave | **`my_models/robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt`** | **{b}** | **{n8}** | — | **{delta}** | — | Wave1 **DONE** · `…/turn_off_microwave/summary_literal5.json` · lock `…_topk_lock.txt` |\n",
        text2,
        count=1,
    )
    # status blurb
    text2 = re.sub(
        r"microwave base 5/5 / BoN running \(1/5\)|microwave base running|BoN \*\*RUNNING\*\* \(1/5\)",
        "microwave Wave1 **DONE**",
        text2,
    )
    if n or n2:
        res.write_text(text2, encoding="utf-8")
        print(f"RESULTS.md patched (main={n}, detail={n2})")
    else:
        print("WARN: RESULTS.md patterns not matched — numbers still in HF README")
    # also drop a snippet for local sync
    snip = root / "logs/mw_wave1_tablep_snippet.md"
    snip.write_text(
        f"| turn_off_microwave | **ep-0500 @0.620** lock | **{b}** | **{n8}** | — | **{delta}** | — | Wave1 **DONE** |\n",
        encoding="utf-8",
    )
    print(f"wrote {snip}")
PY

# --- upload ---
log "uploading staging → ${HF_REPO}"
resolve_hf
# upload folder contents (README + robocasa/)
"${HF_BIN}" upload "${HF_REPO}" "${STAGE}" . --repo-type model 2>&1 | tee -a "${LOG}"
log "=== HF upload DONE ${HF_REPO} ==="
log "URL: https://huggingface.co/${HF_REPO}"
