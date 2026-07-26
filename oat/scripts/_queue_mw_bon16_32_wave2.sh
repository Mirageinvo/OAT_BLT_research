#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LOG="logs/mw_bon16_32_queue_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
echo "=== QUEUE wave2 (host) | $(date -Iseconds) ===" | tee "${LOG}"

chain_done() {
  local key="$1"
  if grep -q "CHAIN DONE" logs/mw_${key}_bon16_32_*.log 2>/dev/null; then
    return 0
  fi
  if tmux has-session -t "mw_${key}_bon16_32" 2>/dev/null; then
    return 1
  fi
  return 0
}

wait_chain() {
  local key="$1"
  echo "[wait] ${key} $(date -Iseconds)" | tee -a "${LOG}"
  while ! chain_done "${key}"; do
    echo "[wait] ${key} still running $(date -Iseconds)" | tee -a "${LOG}"
    sleep 60
  done
  echo "[ready] ${key} $(date -Iseconds)" | tee -a "${LOG}"
}

wait_chain stick
wait_chain coffee

echo "[launch] disassemble + box-close $(date -Iseconds)" | tee -a "${LOG}"
tmux kill-session -t mw_disassemble_bon16_32 2>/dev/null || true
tmux kill-session -t mw_box_close_bon16_32 2>/dev/null || true

tmux new-session -d -s mw_disassemble_bon16_32 \
  "docker exec -w /workspace/oat oat_mipt_robomimic_askhabaliev_gs bash scripts/_run_mw_tablep_bon16_32.sh disassemble 1 output/20260711/134440_train_oatpolicy_mw-disassemble_st_N50/checkpoints/ep-1400_sr-0.700.ckpt; echo EXIT=\$?; exec bash"

tmux new-session -d -s mw_box_close_bon16_32 \
  "docker exec -w /workspace/oat oat_mipt_robomimic_askhabaliev_gs bash scripts/_run_mw_tablep_bon16_32.sh box-close 0 output/20260711/134439_train_oatpolicy_mw-box-close_st_N50/checkpoints/ep-2000_sr-0.552.ckpt; echo EXIT=\$?; exec bash"

echo "[launched] wave2 $(date -Iseconds)" | tee -a "${LOG}"
