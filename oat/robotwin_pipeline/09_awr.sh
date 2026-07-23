#!/usr/bin/env bash
# 09 — AWR distillation: collect BoN rollouts -> weighted SFT of the AR head -> eval single-forward.
# NOTE: collect_awr_dataset.py imports LIBERO env helpers; add a RoboTwin branch/copy first
#       (see ROBOTWIN_TRAIN_GUIDE §7) OR it will fail to build the env.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

echo "[09a] collect BoN-distillation dataset"
${OATPY} scripts/collect_awr_dataset.py -c "${POLICY_CKPT}" \
  -o "my_datasets/rt_awr_${TASK}.npz" --n_chunks 20000 --n_tasks 1 --bon_n 8 --n_workers 6

echo "[09b] train AWR"
${OATPY} scripts/train_awr.py -i "my_datasets/rt_awr_${TASK}.npz" \
  -c "${POLICY_CKPT}" -o "${AWR_CKPT}" --beta 0.5 --beta_kl 0.05 --epochs 100

echo "[09c] eval AWR (single-sample)"
${OATPY} scripts/eval_policy_sim.py -c "${AWR_CKPT}" -o eval_out/rt_awr -n 3 \
  --entropy_threshold 0 --use_k_tokens 8
echo "[09] done. Report: base / BoN N4-8-16 / AWR -> send to PAPER_MASTER.md §3.3."
