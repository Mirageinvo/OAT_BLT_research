#!/usr/bin/env bash
# Table C′ fill order (user): RoboCasa Singles ALL first → then BoNs ALL → then RM+MW BoN16/32.
# AWR16 skipped. Locked latency_fair_kv.json never overwritten.
#
# Trigger: wait until current in-flight Wave1 seed writes eval_log, pause THAT suite,
# run latency on its GPU, resume Wave1. Other suite keeps running.
set -euo pipefail
cd /workspace/oat
mkdir -p logs
LOG=logs/watch_wave1_then_latency.log

SINK_SEED=output/eval/matched_s10000/robocasa/turn_off_sink_faucet/bon_n8_seed10002/eval_log.json
MW_SEED=output/eval/matched_s10000/robocasa/turn_off_microwave/baseline_seed10003/eval_log.json

RC_ALL="coffee_press_button close_drawer turn_off_sink_faucet turn_off_microwave"
RM_MW="can lift square coffee-pull stick-pull disassemble box-close"

declare -A RC_CKPT=(
  [coffee_press_button]=/workspace/oat/my_models/robocasa_coffee_press_button_topk_ep0500_sr0.600.ckpt
  [close_drawer]=/workspace/oat/my_models/robocasa_close_drawer_topk_ep0500_sr0.700.ckpt
  [turn_off_sink_faucet]=/workspace/oat/my_models/robocasa_turn_off_sink_faucet_topk_ep0500_sr0.580.ckpt
  [turn_off_microwave]=/workspace/oat/my_models/robocasa_turn_off_microwave_topk_ep0500_sr0.620.ckpt
)
declare -A RC_GPU=(
  [turn_off_sink_faucet]=1
  [turn_off_microwave]=0
)

log() { echo "$*" | tee -a "${LOG}"; }

find_wave1_bash() {
  local want="$1"
  for p in $(pgrep -f 'cluster_robocasa_literal5_wave1.sh' || true); do
    envf="/proc/${p}/environ"
    [[ -r "${envf}" ]] || continue
    suite=$(tr '\0' '\n' < "${envf}" | awk -F= '/^SUITE=/{print $2}')
    if [[ "${suite}" == "${want}" ]]; then echo "${p}"; return 0; fi
  done
  return 1
}

pause_suite() {
  local suite="$1"
  local bash_pid
  bash_pid=$(find_wave1_bash "${suite}" || true)
  [[ -n "${bash_pid}" ]] || { log "[watch] no wave1 bash for ${suite}"; return 0; }
  log "[watch] pause ${suite} bash=${bash_pid}"
  pkill -P "${bash_pid}" 2>/dev/null || true
  kill "${bash_pid}" 2>/dev/null || true
  sleep 2
  pkill -f "matched_s10000/robocasa/${suite}/" 2>/dev/null || true
  sleep 2
}

resume_suite() {
  local suite="$1" gpu="$2" ckpt="$3"
  log "[watch] resume wave1 ${suite} gpu=${gpu}"
  nohup env SUITE="${suite}" BASE_CKPT="${ckpt}" GPU="${gpu}" \
    bash scripts/cluster_robocasa_literal5_wave1.sh \
    >> "logs/rc_wave1_${suite}_literal5_resumed.log" 2>&1 &
  log "[watch] resumed PID=$!"
}

if [[ -z "${OAT_GIT_COMMIT:-}" ]]; then
  log "ERROR: OAT_GIT_COMMIT unset"; exit 1
fi
export OAT_GIT_BRANCH="${OAT_GIT_BRANCH:-robocasa}" OAT_GIT_DIRTY="${OAT_GIT_DIRTY:-1}"

log "=== C′ order: RC Single→RC BoN→RM/MW BoN16/32 $(date -Iseconds) ==="
log "waiting for ${SINK_SEED} OR ${MW_SEED}"

freed_suite=""; freed_gpu=""; freed_ckpt=""
while true; do
  if [[ -f "${SINK_SEED}" ]]; then
    freed_suite=turn_off_sink_faucet; freed_gpu=1
    freed_ckpt=${RC_CKPT[$freed_suite]}; break
  fi
  if [[ -f "${MW_SEED}" ]]; then
    freed_suite=turn_off_microwave; freed_gpu=0
    freed_ckpt=${RC_CKPT[$freed_suite]}; break
  fi
  sleep 20
done

log "[watch] ${freed_suite} seed DONE — pause + latency on GPU ${freed_gpu}"
pause_suite "${freed_suite}"

# 1) ALL RoboCasa Singles (table order), merge into latency_fair_kv_n16.json
log "[latency] Phase 1/3: RC Single ×4"
GPU="${freed_gpu}" BON_NS="" INCLUDE_SINGLE=1 VENV=.venv_robocasa \
  SUITES="${RC_ALL}" \
  bash scripts/cluster_latency_cprime_no_awr.sh 2>&1 | tee -a "${LOG}"

# 2) ALL RoboCasa BoN 8/16/32
log "[latency] Phase 2/3: RC BoN8/16/32 ×4"
GPU="${freed_gpu}" BON_NS=8,16,32 INCLUDE_SINGLE=0 VENV=.venv_robocasa \
  SUITES="${RC_ALL}" \
  bash scripts/cluster_latency_cprime_no_awr.sh 2>&1 | tee -a "${LOG}"

# 3) RM+MW BoN16/32 (locked Single/BoN8/AWR8 untouched)
log "[latency] Phase 3/3: RM+MW BoN16/32"
GPU="${freed_gpu}" BON_NS=16,32 INCLUDE_SINGLE=0 VENV=.venv \
  SUITES="${RM_MW}" \
  bash scripts/cluster_latency_cprime_no_awr.sh 2>&1 | tee -a "${LOG}"

resume_suite "${freed_suite}" "${freed_gpu}" "${freed_ckpt}"
log "=== C′ latency phases DONE $(date -Iseconds); Wave1 resumed ==="
