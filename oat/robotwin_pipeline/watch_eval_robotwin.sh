#!/usr/bin/env bash
# External SR watcher for a RUNNING RoboTwin policy train. Every EVERY epochs it runs a path-B
# BASELINE eval (N=1) on the newest checkpoint and appends epoch,SR to sr_curve.csv.
#
# SAFE for training: separate process, only READS/COPIES checkpoints, evals in a subprocess.
# It cannot crash the training. Worst case (shared GPU OOM) = that one eval fails (SR=NA) and
# training continues untouched.
#
# Run alongside 06_train_policy.sh (same TASK env):
#   TASK=beat_block_hammer TASK_CFG=beat_block_hammer NDEMO=330 \
#     EVERY=75 TEST_NUM=15 EVAL_GPU=1 \
#     nohup bash robotwin_pipeline/watch_eval_robotwin.sh > ~/watch_beat.log 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
set +e   # never let a failed eval kill the loop

EVERY="${EVERY:-75}"
TEST_NUM="${TEST_NUM:-15}"
EVAL_GPU="${EVAL_GPU:-1}"
SEED="${SEED:-0}"
POLL="${POLL:-60}"
RUN_DIR="${RUN_DIR:-$(ls -dt ${OAT_DIR}/output/*/*oatpolicy_${TASK}* 2>/dev/null | head -1)}"
: "${RUN_DIR:?no policy run dir for TASK=${TASK}; pass RUN_DIR=...}"
CKDIR="${RUN_DIR}/checkpoints"
CSV="${RUN_DIR}/sr_curve.csv"

# install/refresh the OAT adapter into RoboTwin + set eval test_num (baseline path B)
mkdir -p "${ROBOTWIN_DIR}/policy/OAT"
cp -f "${OAT_DIR}/robotwin_pipeline/policy_OAT/"* "${ROBOTWIN_DIR}/policy/OAT/"
chmod +x "${ROBOTWIN_DIR}/policy/OAT/eval.sh"
sed -i -E "s/test_num = [0-9]+/test_num = ${TEST_NUM}/" "${ROBOTWIN_DIR}/script/eval_policy.py"
grep -n "test_num =" "${ROBOTWIN_DIR}/script/eval_policy.py" | head -1

[ -f "${CSV}" ] || echo "epoch,success_rate,ckpt,time" > "${CSV}"
echo "[watch] RUN_DIR=${RUN_DIR}"
echo "[watch] every=${EVERY}ep  test_num=${TEST_NUM}  gpu=${EVAL_GPU}  seed=${SEED}  csv=${CSV}"
echo "[watch] NB baseline-only, N=1. SR here is a rough in-training signal (eval seeds may overlap"
echo "        collection 0..600 -> optimistic); the paper number uses 10_eval_pathB.sh + shifted seeds."

last=-1
while true; do
  ck=$(ls -t "${CKDIR}"/ep-*.ckpt 2>/dev/null | head -1)
  if [ -n "${ck}" ]; then
    ep=$(basename "${ck}" | grep -oE 'ep-[0-9]+' | grep -oE '[0-9]+')
    ep=$((10#${ep}))
    if [ "${ep}" -ge "$((last + EVERY))" ]; then
      snap="/tmp/watch_${TASK}.ckpt"
      cp -f "${ck}" "${snap}"    # freeze so topk can't delete it mid-eval
      echo "[watch] $(date +%H:%M:%S) eval ep=${ep} ckpt=$(basename ${ck})"
      evlog="/tmp/watch_eval_${TASK}.out"    # live-tailable: tail -f this file during an eval
      ( cd "${ROBOTWIN_DIR}/policy/OAT" && \
        OAT_DIR="${OAT_DIR}" CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
        bash eval.sh "${TASK}" "${ROBOTWIN_CONFIG}" "${snap}" 1 "${SEED}" "${EVAL_GPU}" ) > "${evlog}" 2>&1
      sr=$(grep -i "Success rate" "${evlog}" | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?%' | tail -1)
      [ -z "${sr}" ] && sr="NA"
      echo "${ep},${sr},$(basename ${ck}),$(date +%H:%M:%S)" >> "${CSV}"
      echo "[watch] -> ep=${ep} SR=${sr}   (logged to ${CSV})"
      last=${ep}
    fi
  fi
  sleep "${POLL}"
done
