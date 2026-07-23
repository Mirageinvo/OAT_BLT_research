#!/usr/bin/env bash
# 10 — RoboTwin eval via THEIR harness (path B): baseline (single) + best-of-N (vote).
# Replaces our 07/08 for RoboTwin (our AsyncVectorEnv runner can't fork under SAPIEN).
# Uses the OAT policy adapter (robotwin_pipeline/policy_OAT) copied into RoboTwin/policy/OAT.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
set +e

: "${POLICY_CKPT:?set POLICY_CKPT in config.sh to the trained policy checkpoint first}"
BON_N="${BON_N:-8}"      # best-of-N to test (also runs baseline N=1)
GPU="${GPU:-0}"
SEED="${SEED:-0}"

# 1) install/refresh the OAT adapter inside RoboTwin
mkdir -p "${ROBOTWIN_DIR}/policy/OAT"
cp -f "$(dirname "${BASH_SOURCE[0]}")/policy_OAT/"* "${ROBOTWIN_DIR}/policy/OAT/"
chmod +x "${ROBOTWIN_DIR}/policy/OAT/eval.sh"

cd "${ROBOTWIN_DIR}/policy/OAT"
echo "[10] BASELINE (N=1)  ckpt=${POLICY_CKPT}"
OAT_DIR="${OAT_DIR}" bash eval.sh "${TASK}" "${ROBOTWIN_CONFIG}" "${POLICY_CKPT}" 1 "${SEED}" "${GPU}"

echo "[10] BEST-OF-N (N=${BON_N})"
OAT_DIR="${OAT_DIR}" bash eval.sh "${TASK}" "${ROBOTWIN_CONFIG}" "${POLICY_CKPT}" "${BON_N}" "${SEED}" "${GPU}"

echo "[10] done. RoboTwin prints 'Success rate: suc/test_num => X%' for each. Record baseline vs BoN."
