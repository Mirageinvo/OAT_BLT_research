#!/usr/bin/env bash
# ccmplanner.mipt.ru workflow (Docker GPU, no SLURM). Run AFTER sync_to_cluster.sh from your Mac.
#
#   ssh -i ~/.ssh/mipt_lab askhabaliev_gs@100.98.148.137
#   cd ~/mipt_paper/oat
#   bash scripts/cluster_ccmplanner.sh setup          # docker + uv sync (once)
#   bash scripts/cluster_ccmplanner.sh tok            # phase C (fresh)
#   bash scripts/cluster_ccmplanner.sh tok_resume     # continue until MSE ~0.002
#   export TOKENIZER_CKPT=output/.../ep-xxxx_mse-0.002.ckpt
#   bash scripts/cluster_ccmplanner.sh policy         # phase D
#   bash scripts/cluster_ccmplanner.sh status         # logs / best ckpts
#
# Runs inside container oat_mipt_robomimic_${USER} when possible.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

CONTAINER="${CONTAINER:-oat_mipt_robomimic_${USER}}"
IMAGE="${IMAGE:-x86_64/oat-robomimic:latest}"
HOST_OAT="${HOST_OAT:-${HOME}/mipt_paper/oat}"
ACTION="${1:-status}"

ensure_container() {
  if docker ps -q -f "name=^${CONTAINER}$" | grep -q .; then
    return 0
  fi
  if docker ps -aq -f "name=^${CONTAINER}$" | grep -q .; then
    echo "Starting ${CONTAINER}"
    docker start "${CONTAINER}"
    return 0
  fi
  echo "Creating ${CONTAINER}"
  docker run -d \
    --name "${CONTAINER}" \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -e MUJOCO_GL=egl \
    --ipc host \
    -v "${HOST_OAT}:/workspace/oat" \
    -w /workspace/oat \
    "${IMAGE}" \
    sleep infinity
}

in_container() {
  ensure_container
  docker exec "${CONTAINER}" bash -lc "$1"
}

run_oat() {
  in_container "
    set -euo pipefail
    export PATH=\"\${HOME}/.local/bin:\${PATH}\"
    cd /workspace/oat
    if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
    $1
  "
}

setup() {
  in_container "
    set -euo pipefail
    export PATH=\"\${HOME}/.local/bin:\${PATH}\"
    cd /workspace/oat
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH=\"\${HOME}/.local/bin:\${PATH}\"
    uv sync
    source .venv/bin/activate
    bash scripts/cluster_ensure_v100_torch.sh
    python -c \"
from robomimic.models.obs_core import VisualCore
from oat.perception.robomimic_vision_encoder import RobomimicRgbEncoder
import robomimic, robosuite
print('robomimic', robomimic.__version__, 'robosuite', robosuite.__version__)
print('VisualCore + RobomimicRgbEncoder OK')
\"
  "
}

status() {
  echo "=== container ==="
  docker ps -f "name=${CONTAINER}" --format '{{.Names}} {{.Status}}'
  run_oat "
    cd /workspace/oat
    echo '--- tokenizer checkpoints (best MSE) ---'
    find output -name 'ep-*_mse-*.ckpt' 2>/dev/null | sort -t- -k3 -n | tail -5 || true
    echo '--- policy checkpoints ---'
    find output -name 'ep-*_sr-*.ckpt' 2>/dev/null | tail -5 || true
    echo '--- recent logs ---'
    tail -8 logs/pipeline_tok.log 2>/dev/null || true
    tail -8 logs/train_policy_lift.log 2>/dev/null || true
  "
}

tok_bg() {
  mkdir -p logs
  in_container "
    export PATH=\"\${HOME}/.local/bin:\${PATH}\"
    cd /workspace/oat
    source .venv/bin/activate
    nohup bash scripts/cluster_ccmplanner.sh _tok_inner > logs/tok_nohup.out 2>&1 &
    echo \"tokenizer PID \$!\"
  "
}

_tok_inner() {
  export PATH="${HOME}/.local/bin:${PATH}"
  cd /workspace/oat
  source .venv/bin/activate
  NUM_PROCESSES="${NUM_PROCESSES:-1}" bash scripts/prepare_robomimic_lift.sh tok
}

_tok_resume_inner() {
  export PATH="${HOME}/.local/bin:${PATH}"
  cd /workspace/oat
  source .venv/bin/activate
  TOK_RUN_DIR="${TOK_RUN_DIR:?Set TOK_RUN_DIR=output/YYYYMMDD/HHMMSS_train_oattok_lift_N200}"
  [[ -d "${TOK_RUN_DIR}/checkpoints" ]] || { echo "No checkpoints in ${TOK_RUN_DIR}"; exit 1; }
  echo "Resume tokenizer in ${TOK_RUN_DIR}"
  HYDRA_FULL_ERROR=1 accelerate launch \
    --num_machines 1 --multi_gpu --num_processes "${NUM_PROCESSES:-1}" \
    scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=robomimic/lift \
    training.num_demo=200 \
    training.resume=true \
    checkpoint.topk.k=3 \
    logging.mode=disabled \
    hydra.run.dir="${TOK_RUN_DIR}"
}

policy_tmux() {
  mkdir -p logs
  TOKENIZER_CKPT="${TOKENIZER_CKPT:?export TOKENIZER_CKPT}"
  NUM_PROCESSES="${NUM_PROCESSES:-2}"
  ROLLOUT_EVERY="${ROLLOUT_EVERY:-100}"
  SESSION="${POLICY_TMUX_SESSION:-policy_lift}"
  in_container "
    export PATH=\"\${HOME}/.local/bin:\${PATH}\"
    cd /workspace/oat
    tmux kill-session -t ${SESSION} 2>/dev/null || true
    tmux new-session -d -s ${SESSION} bash -lc '
      set -euo pipefail
      export PATH=\"\${HOME}/.local/bin:\${PATH}\"
      cd /workspace/oat
      source .venv/bin/activate
      export OAT_USE_UV_RUN=0
      export TOKENIZER_CKPT=\"${TOKENIZER_CKPT}\"
      export NUM_PROCESSES=${NUM_PROCESSES}
      export NUM_DEMO=${NUM_DEMO:-200}
      export ROLLOUT_EVERY=${ROLLOUT_EVERY}
      export POLICY_PROFILE=${POLICY_PROFILE:-aggressive}
      export MUJOCO_GL=egl
      if [[ -d \"\${HOME}/.mujoco/mujoco210/bin\" ]]; then
        export LD_LIBRARY_PATH=\"\${HOME}/.mujoco/mujoco210/bin:\${LD_LIBRARY_PATH:-}\"
      fi
      bash scripts/cluster_ensure_v100_torch.sh
      : > logs/train_policy_lift.log
      bash scripts/prepare_robomimic_lift.sh policy 2>&1 | tee -a logs/train_policy_lift.log
    '
    echo \"tmux session: ${SESSION} (attach: tmux attach -t ${SESSION})\"
    sleep 2
    tmux list-sessions | grep ${SESSION} || { echo FAILED; exit 1; }
  "
}

policy_bg() {
  mkdir -p logs
  TOKENIZER_CKPT="${TOKENIZER_CKPT:?export TOKENIZER_CKPT}"
  NUM_PROCESSES="${NUM_PROCESSES:-1}"
  ROLLOUT_EVERY="${ROLLOUT_EVERY:-100}"
  in_container "
    export PATH=\"\${HOME}/.local/bin:\${PATH}\"
    cd /workspace/oat
    source .venv/bin/activate
    export OAT_USE_UV_RUN=0
    export TOKENIZER_CKPT='${TOKENIZER_CKPT}'
    export NUM_PROCESSES='${NUM_PROCESSES}'
    export NUM_DEMO='${NUM_DEMO:-200}'
    export ROLLOUT_EVERY='${ROLLOUT_EVERY}'
  nohup bash scripts/prepare_robomimic_lift.sh policy > logs/train_policy_lift_nohup.out 2>&1 &
    echo \"policy training PID \$!\"
  "
}

policy_fg() {
  run_oat "
    cd /workspace/oat
    export OAT_USE_UV_RUN=0
    export TOKENIZER_CKPT=\"${TOKENIZER_CKPT:?export TOKENIZER_CKPT}\"
    export NUM_PROCESSES=\"${NUM_PROCESSES:-1}\"
    export NUM_DEMO=\"${NUM_DEMO:-200}\"
    export ROLLOUT_EVERY=\"${ROLLOUT_EVERY:-100}\"
    bash scripts/prepare_robomimic_lift.sh policy
  "
}

case "${ACTION}" in
  setup) setup ;;
  status) status ;;
  tok) tok_bg ;;
  _tok_inner) _tok_inner ;;
  tok_resume)
    TOK_RUN_DIR="${TOK_RUN_DIR:-output/20260704/203215_train_oattok_lift_N200}"
    export TOK_RUN_DIR NUM_PROCESSES="${NUM_PROCESSES:-1}"
    run_oat "bash scripts/cluster_ccmplanner.sh _tok_resume_inner"
    ;;
  _tok_resume_inner) _tok_resume_inner ;;
  policy) policy_fg ;;
  policy_bg) policy_bg ;;
  policy_tmux) policy_tmux ;;
  *)
    echo "Usage: $0 {setup|status|tok|tok_resume|policy|policy_bg|policy_tmux}"
    exit 1
    ;;
esac
