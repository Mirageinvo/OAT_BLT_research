#!/usr/bin/env bash
# 07 — baseline eval (single-sample, full budget) + HEADROOM GATE.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
echo "[07] baseline eval ${POLICY_CKPT}"
uv run scripts/eval_policy_sim.py -c "${POLICY_CKPT}" -o eval_out/rt_base -n 3 \
  --entropy_threshold 0 --use_k_tokens 8
echo "[07] done. GATE: mean_success_rate must be > ~0.2 to have BoN headroom."
echo "     If SR is on the floor -> more demos / a simpler task; else run 08."
