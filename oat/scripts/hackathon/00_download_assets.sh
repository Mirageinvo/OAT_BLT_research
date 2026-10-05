#!/usr/bin/env bash
# Download paper baseline assets from HF: tokenizers, policies, zarr, eval summaries, demo videos.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL_REPO="hackhackhack66666/robomimic-oattok-policy"
DATA_REPO="hackhackhack66666/robomimic_zarr"
EVAL_REPO="hackhackhack66666/aaai27-models"
STAGE="${STAGE:-hackathon_assets}"
OUT="${OUT:-hackathon_output}"
DOWNLOAD_MEDIA="${DOWNLOAD_MEDIA:-1}"
MEDIA_PER_TASK="${MEDIA_PER_TASK:-5}"

mkdir -p "${STAGE}/tokenizers" "${STAGE}/policies" \
  data/robomimic/hdf5_datasets \
  "${OUT}/eval/matched_s10000" "${OUT}/demo/media" "${OUT}/hydra"

download_ckpt() {
  local repo="$1" path="$2" dest="$3"
  mkdir -p "$(dirname "${dest}")"
  hf download "${repo}" "${path}" --local-dir "$(dirname "${dest}")/_dl"
  cp "$(dirname "${dest}")/_dl/${path}" "${dest}"
  echo "  -> ${dest}"
}

echo "=== Tokenizers + policies (robomimic-oattok-policy) ==="
while read -r task tok pol; do
  download_ckpt "${MODEL_REPO}" "${tok}" "${STAGE}/tokenizers/${task}.ckpt"
  download_ckpt "${MODEL_REPO}" "${pol}" "${STAGE}/policies/${task}.ckpt"
done <<'EOF'
lift tokenizers/lift/ep-1970_mse-0.006.ckpt policies/lift/ep-0900_sr-0.930.ckpt
can tokenizers/can/ep-0520_mse-0.005.ckpt policies/can/ep-1700_sr-0.940.ckpt
square tokenizers/square/ep-0690_mse-0.004.ckpt policies/square/ep-0700_sr-0.420.ckpt
EOF

echo "=== Zarr datasets (robomimic_zarr) ==="
mkdir -p data/robomimic
for z in lift_N200 can_N200 square_N200; do
  if [[ -d "data/robomimic/${z}.zarr" ]]; then
    echo "  skip data/robomimic/${z}.zarr (exists)"
    continue
  fi
  hf download "${DATA_REPO}" \
    --repo-type dataset \
    --local-dir data/robomimic \
    --include "${z}.zarr/**"
  echo "  data/robomimic/${z}.zarr"
done

echo "=== Paper eval summaries + hydra configs (aaai27-models) — optional ==="
if hf download "${EVAL_REPO}" "eval/matched_s10000/lift/summary.json" \
    --local-dir "${OUT}" 2>/dev/null; then
  for task in lift can square; do
    hf download "${EVAL_REPO}" "eval/matched_s10000/${task}/summary.json" \
      --local-dir "${OUT}" 2>/dev/null || true
  done
else
  echo "  skip aaai27-models (auth optional for train)"
fi

if [[ "${DOWNLOAD_MEDIA}" == "1" ]]; then
  echo "=== Demo videos (first ${MEDIA_PER_TASK} mp4 per task) ==="
  python3 scripts/hackathon/download_demo_media.py \
    --repo "${EVAL_REPO}" \
    --manifest scripts/hackathon/baseline_manifest.json \
    --output_dir "${OUT}/demo/media" \
    --per_task "${MEDIA_PER_TASK}"
fi

echo "=== HDF5 env metadata (for sim) — aaai-datasets ==="
DATA_HF="${DATA_HF:-hackhackhack66666/aaai-datasets}"
for h5 in lift_mh_image can_mh_image; do
  dest="data/robomimic/hdf5_datasets/${h5}.hdf5"
  if [[ -f "${dest}" ]]; then
    echo "  skip ${dest} (exists)"
    continue
  fi
  echo "  downloading ${h5}.hdf5 from ${DATA_HF} ..."
  hf download "${DATA_HF}" "robomimic/hdf5/${h5}.hdf5" \
    --repo-type dataset \
    --local-dir data/robomimic/hdf5_datasets
  nested="data/robomimic/hdf5_datasets/robomimic/hdf5/${h5}.hdf5"
  if [[ -f "${nested}" && ! -f "${dest}" ]]; then
    ln -sf "robomimic/hdf5/${h5}.hdf5" "${dest}"
  fi
  [[ -f "${dest}" ]] || { echo "  ERROR: ${dest} missing after download"; exit 1; }
  echo "  -> ${dest}"
done
# square optional (not needed for 2-task POC)
if [[ "${DOWNLOAD_SQUARE_HDF5:-0}" == "1" ]]; then
  h5=square_mh_image
  dest="data/robomimic/hdf5_datasets/${h5}.hdf5"
  if [[ ! -f "${dest}" ]]; then
    hf download "${DATA_HF}" "robomimic/hdf5/${h5}.hdf5" \
      --repo-type dataset --local-dir data/robomimic/hdf5_datasets
    ln -sf "robomimic/hdf5/${h5}.hdf5" "${dest}" 2>/dev/null || true
  fi
fi

echo ""
echo "Done."
echo "  Policies:  ${STAGE}/policies/{lift,can,square}.ckpt"
echo "  Tokenizers:${STAGE}/tokenizers/{lift,can,square}.ckpt"
echo "  Eval:      ${OUT}/eval/matched_s10000/*/summary.json"
echo "  Demo mp4:  ${OUT}/demo/media/"
echo "Next: bash scripts/hackathon/01_eval_baseline.sh   # optional reproduce SR"
echo "      bash scripts/hackathon/02_prepare_demo.sh    # mp4 -> gif for slides"
