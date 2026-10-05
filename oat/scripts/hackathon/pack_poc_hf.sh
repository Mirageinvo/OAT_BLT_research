#!/usr/bin/env bash
# Pack Skoltech POC artifacts for HF repo hackhackhack66666/Skoltech-quickSuccess
set -euo pipefail
cd "$(dirname "$0")/../.."

export OAT_USE_UV_RUN=0 MUJOCO_GL=egl
PACK="${PACK_DIR:-hackathon_output/hf_pack}"
LIFT_RUN="${LIFT_RUN:-output/20260902/100353_train_oatpolicy_lift_N200}"
CAN_RUN="${CAN_RUN:-output/20260902/100352_train_oatpolicy_can_N200}"
LIFT_CKPT="${LIFT_CKPT:-hackathon_output/policies/poc/lift_latest.ckpt}"
CAN_CKPT="${CAN_CKPT:-hackathon_output/policies/poc/can_latest.ckpt}"
NUM_EXP="${NUM_EXP:-10}"
SKIP_MULTI="${SKIP_MULTI:-0}"

mkdir -p "${PACK}/policies" "${PACK}/gifs" "${PACK}/eval/lift" "${PACK}/eval/can"

echo "=== copy checkpoints ==="
cp -f "${LIFT_CKPT}" "${PACK}/policies/lift_poc_latest.ckpt"
cp -f "${CAN_CKPT}" "${PACK}/policies/can_poc_latest.ckpt"

echo "=== copy gifs + quick eval logs ==="
cp -f hackathon_output/gifs/lift_success_00_seed10000.gif "${PACK}/gifs/" 2>/dev/null || true
cp -f hackathon_output/gifs/can_success_00_seed10000.gif "${PACK}/gifs/" 2>/dev/null || true
cp -f hackathon_output/eval/timed/lift/eval_log.json "${PACK}/eval/lift/quick_eval_log.json" 2>/dev/null || true
cp -f hackathon_output/eval/timed/can/eval_log.json "${PACK}/eval/can/quick_eval_log.json" 2>/dev/null || true

echo "=== training dashboards ==="
python scripts/hackathon/make_poc_dashboard.py --mode training --task lift --run_dir "${LIFT_RUN}" -o "${PACK}"
python scripts/hackathon/make_poc_dashboard.py --mode training --task can --run_dir "${CAN_RUN}" -o "${PACK}"

if [[ "${SKIP_MULTI}" != "1" ]]; then
  echo "=== multi-exp eval for dashboard (num_exp=${NUM_EXP}) ==="
  python scripts/hackathon/run_eval_multiexp.py -c "${LIFT_CKPT}" --task lift -n "${NUM_EXP}" \
    -o "${PACK}/eval/lift/per_exp.json" --gpu 0 &
  PID_L=$!
  python scripts/hackathon/run_eval_multiexp.py -c "${CAN_CKPT}" --task can -n "${NUM_EXP}" \
    -o "${PACK}/eval/can/per_exp.json" --gpu 1 &
  PID_C=$!
  wait "${PID_L}" "${PID_C}"
fi

echo "=== eval dashboards ==="
python scripts/hackathon/make_poc_dashboard.py --mode eval --task lift \
  --per_exp_json "${PACK}/eval/lift/per_exp.json" -o "${PACK}"
python scripts/hackathon/make_poc_dashboard.py --mode eval --task can \
  --per_exp_json "${PACK}/eval/can/per_exp.json" -o "${PACK}"

echo "=== MANIFEST ==="
python3 - <<'PY' "${PACK}"
import json, pathlib, sys
pack = pathlib.Path(sys.argv[1])
manifest = {"tasks": {}}
for task in ("lift", "can"):
    ts = pack / "logs" / task / "training_summary.json"
    es = pack / "eval" / task / "eval_dashboard_summary.json"
    row = {}
    if ts.is_file():
        row["training"] = json.loads(ts.read_text())
    if es.is_file():
        row["eval"] = json.loads(es.read_text())
    manifest["tasks"][task] = row
(pack / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY

echo "Packed → ${PACK}"
