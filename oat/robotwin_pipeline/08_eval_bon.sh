#!/usr/bin/env bash
# 08 — verifier-free best-of-N sweep (vote), no training. The core positive result.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
for N in 4 8 16; do
  echo "[08] BoN N=${N}"
  ${OATPY} scripts/eval_policy_sim.py -c "${POLICY_CKPT}" -o "eval_out/rt_bon${N}" -n 3 \
    --bon_free ${N} --bon_signal vote --use_k_tokens 8
done
echo "[08] done. Record SR(base), SR(N=4/8/16). Expect monotone rise (compounding). Then 09."
