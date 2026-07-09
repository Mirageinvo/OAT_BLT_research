#!/usr/bin/env bash
# Aggressive profile validation — matches the validator verdict table.
#
# CLOSED (code + live aggressive train): fp16/AMP, SR double-count guard, OOM, DDP ckpt.
# RUN ONCE:  ab_parallel — same ckpt, SR@2 vs SR@8 (metric confirm; 8 already from train).
# MONITOR:   SR@100/200 on aggressive train (batch/LR dynamics — NOT an env A/B).
# OPTIONAL:  profile_sr — only if SR curve looks wrong vs paper / you want hard A/B on hyperparams.
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

RUN_DIR="${RUN_DIR:-$(ls -dt output/*/*_train_oatpolicy_lift_N200 2>/dev/null | head -1 || true)}"
RUN_DIR_AGGR="${RUN_DIR_AGGR:-${RUN_DIR}}"
RUN_DIR_DEFAULT="${RUN_DIR_DEFAULT:-}"
ACTION="${1:-status}"

_mujoco() {
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  if [[ -d "${HOME}/.mujoco/mujoco210/bin" ]]; then
    export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
  fi
}

_sr_curve() {
  python - <<'PY' "${1}"
import json, sys
path = sys.argv[1]
try:
    rows = [json.loads(l) for l in open(path) if l.strip()]
except FileNotFoundError:
    raise SystemExit(0)
for r in rows:
    if "mean_success_rate" in r:
        print(f"{r.get('epoch', '?'):>6}  {r['mean_success_rate']:.4f}")
PY
}

_train_sr_at_epoch() {
  python - <<'PY' "${1}" "${2:-0}"
import json, sys
path, ep = sys.argv[1], int(sys.argv[2])
for line in open(path):
    r = json.loads(line)
    if r.get("epoch") == ep and "mean_success_rate" in r:
        print(r["mean_success_rate"]); break
else:
    print("")
PY
}

status() {
  echo "=== Validator verdict (aggressive profile) ==="
  echo ""
  echo "| Risk | Status |"
  echo "|------|--------|"
  echo "| fp16 без GradScaler | CLOSED — Accelerate autocast + backward |"
  echo "| SR double-count @ 8 env | CLOSED в коде; ab_parallel подтверждает |"
  echo "| OOM batch 1280 | CLOSED — ~20/32 GB |"
  echo "| DDP checkpoint race | CLOSED — rank0 only |"
  echo "| batch/LR сдвиг SR | OPEN — мониторить SR@100/200 (не env A/B) |"
  echo ""
  echo "--- #1 Epoch 0 gate ---"
  CKPT=$(find "${RUN_DIR_AGGR}/checkpoints" -name 'ep-*_sr-*.ckpt' 2>/dev/null | sort | head -1 || true)
  SR0=$(_train_sr_at_epoch "${RUN_DIR_AGGR}/logs.json" 0 2>/dev/null || true)
  if [[ -n "${CKPT}" && -n "${SR0}" ]]; then
    echo "  PASS: SR@0=${SR0}  ckpt=${CKPT}"
  else
    echo "  WAIT: ckpt=${CKPT:-none}  SR@0=${SR0:-pending}"
  fi
  echo ""
  echo "--- #2 A/B metric (same ckpt, n_parallel_envs 2 vs 8) ---"
  echo "  SR@8 = train eval (aggressive n_parallel_envs=8): ${SR0:-pending}"
  if [[ -f eval_out/ab_parallel_2/eval_log.json ]]; then
    SR2=$(python -c "import json;d=json.load(open('eval_out/ab_parallel_2/eval_log.json'));print(d.get('mean_success_rate_mean',d.get('mean_success_rate')))")
    echo "  SR@2 = standalone eval: ${SR2}"
    echo "  → compare |SR2-SR8| ≲ 0.14"
  elif pgrep -f "eval_policy_sim.py.*ab_parallel_2" >/dev/null 2>&1; then
    echo "  IN PROGRESS"
  else
    echo "  TODO: CKPT=${CKPT:-...} bash scripts/validate_aggressive_profile.sh ab_parallel"
  fi
  echo ""
  echo "--- #3 Smoke loss (#4 VRAM) ---"
  if [[ -f "${RUN_DIR_AGGR}/logs.json" ]]; then
    grep -qi nan "${RUN_DIR_AGGR}/logs.json" 2>/dev/null && echo "  WARNING: NaN in logs" || echo "  OK: no NaN (smoke-train skip)"
    LOSS=$(python -c "import json;r=[json.loads(l) for l in open('${RUN_DIR_AGGR}/logs.json') if l.strip()];print(r[-1].get('train_loss','?'),'@ ep',r[-1].get('epoch','?'))" 2>/dev/null || echo "?")
    echo "  train_loss: ${LOSS}"
  fi
  echo ""
  echo "--- batch/LR: SR curve (monitor @100/200) ---"
  if [[ -f "${RUN_DIR_AGGR}/logs.json" ]]; then
    _sr_curve "${RUN_DIR_AGGR}/logs.json" | sed 's/^/  /'
  fi
  if [[ -n "${RUN_DIR_DEFAULT}" && -f "${RUN_DIR_DEFAULT}/logs.json" ]]; then
    echo "  (optional profile_sr vs default run available)"
  else
    echo "  Footnote if SR differs from paper; profile_sr only if curve looks broken"
  fi
  echo "RUN_DIR=${RUN_DIR_AGGR}"
}

sanity() {
  LOG="${POLICY_LOG:-${RUN_DIR_AGGR}/run_workspace.log}"
  grep -h mean_success_rate "${LOG}" "${RUN_DIR_AGGR}/logs.json" 2>/dev/null | tail -5 || true
  find "${RUN_DIR_AGGR}" -name 'ep-*_sr-*.ckpt' 2>/dev/null | sort | head -3
  _sr_curve "${RUN_DIR_AGGR}/logs.json" 2>/dev/null || true
}

# #2: metric only — SR@2 vs SR@8 on same ckpt (8 from train, don't re-run 8).
ab_parallel() {
  CKPT="${CKPT:?Set CKPT}"
  RUN_DIR="${RUN_DIR:-${RUN_DIR_AGGR}}"
  _mujoco
  [[ -f .venv/bin/activate ]] && source .venv/bin/activate
  export OAT_USE_UV_RUN=0

  SR8=$(_train_sr_at_epoch "${RUN_DIR}/logs.json" 0)
  [[ -z "${SR8}" ]] && SR8=$(python - <<PY
import json
for r in reversed([json.loads(l) for l in open("${RUN_DIR}/logs.json") if l.strip()]):
  if "mean_success_rate" in r: print(r["mean_success_rate"]); break
PY
)
  [[ -n "${SR8}" ]] || { echo "ERROR: no SR in train logs"; exit 1; }

  OUT=eval_out/ab_parallel_2
  if [[ ! -f "${OUT}/eval_log.json" ]]; then
    rm -rf "${OUT}"
    python scripts/eval_policy_sim.py -c "${CKPT}" -o "${OUT}" -n 1 \
      --entropy_threshold 0 --use_k_tokens 8 --n_parallel_envs 2
  fi
  SR2=$(python -c "import json;d=json.load(open('${OUT}/eval_log.json'));print(d.get('mean_success_rate_mean',d.get('mean_success_rate')))")
  DELTA=$(python -c "print(abs(float('${SR2}')-float('${SR8}')))")
  echo ""
  echo "| arm | SR |"
  echo "|-----|-----|"
  echo "| n_parallel_envs=8 (train) | ${SR8} |"
  echo "| n_parallel_envs=2 (eval)  | ${SR2} |"
  echo "|delta| ${DELTA} | (pass if ≤ 0.14)"
  python -c "exit(0 if float('${DELTA}')<=0.14 else 1)" && echo "VERDICT: PASS" || echo "VERDICT: FAIL — runner"
}

ab_eval() { ab_parallel "$@"; }

# OPTIONAL: hard hyperparam A/B — only if SR@100/200 looks wrong.
profile_sr() {
  [[ -n "${RUN_DIR_DEFAULT}" && -f "${RUN_DIR_DEFAULT}/logs.json" ]] || {
    echo "Need RUN_DIR_DEFAULT from POLICY_PROFILE=default train (optional, not primary gate)"
    exit 1
  }
  python - <<'PY' "${RUN_DIR_AGGR}/logs.json" "${RUN_DIR_DEFAULT}/logs.json"
import json, sys
def load(p):
    o = {}
    for line in open(p):
        r = json.loads(line)
        if "mean_success_rate" in r and "epoch" in r:
            o[int(r["epoch"])] = float(r["mean_success_rate"])
    return o
a, d = load(sys.argv[1]), load(sys.argv[2])
for e in sorted(set(a) & set(d)):
    print(f"ep {e}: aggr={a[e]:.3f} default={d[e]:.3f} delta={a[e]-d[e]:+.3f}")
PY
}

vram_peek() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
}

case "${ACTION}" in
  status) status ;;
  sanity) sanity ;;
  ab_parallel|ab_eval) ab_parallel ;;
  profile_sr) profile_sr ;;
  vram_peek) vram_peek ;;
  *) echo "Usage: $0 {status|sanity|ab_parallel|profile_sr|vram_peek}"; exit 1 ;;
esac
