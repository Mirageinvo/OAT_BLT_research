#!/usr/bin/env bash
# Prepare demo gifs for slides: HF baseline videos OR fresh rollouts from baseline ckpts.
set -euo pipefail
cd "$(dirname "$0")/../.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
STAGE="${STAGE:-hackathon_assets}"
MEDIA_DIR="${MEDIA_DIR:-hackathon_output/demo/media}"
GIF_DIR="${GIF_DIR:-hackathon_output/gifs}"
MODE="${MODE:-hf}"   # hf | rollout | both

mkdir -p "${GIF_DIR}"

mp4_to_gif() {
  local mp4="$1" gif="$2"
  ffmpeg -y -loglevel error -i "${mp4}" \
    -vf "fps=10,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
    "${gif}"
}

if [[ "${MODE}" == "hf" || "${MODE}" == "both" ]]; then
  echo "=== GIFs from HF baseline mp4 ==="
  for task in lift can square; do
    d="${MEDIA_DIR}/${task}"
    if [[ ! -d "${d}" ]]; then
      echo "WARN: no ${d} — run 00_download_assets.sh with DOWNLOAD_MEDIA=1"
      continue
    fi
    n=0
    for mp4 in "${d}"/*.mp4; do
      [[ -f "${mp4}" ]] || continue
      stem="$(basename "${mp4}" .mp4)"
      gif="${GIF_DIR}/${task}_baseline_${stem}.gif"
      mp4_to_gif "${mp4}" "${gif}"
      echo "  ${gif}"
      n=$((n + 1))
      [[ "${MODE}" == "hf" && "${n}" -ge 1 ]] && break
    done
  done
fi

if [[ "${MODE}" == "rollout" || "${MODE}" == "both" ]]; then
  echo "=== Fresh success rollouts (baseline ckpts) ==="
  POL_DIR="${STAGE}/policies"
  TARGET="${TARGET:-1}"
  for task in lift can square; do
    ckpt="${POL_DIR}/${task}.ckpt"
    [[ -f "${ckpt}" ]] || { echo "Missing ${ckpt}"; exit 1; }
    uv run python scripts/hackathon/rollout_until_success.py \
      -c "${ckpt}" -o "${GIF_DIR}" --task "${task}" --target "${TARGET}"
  done
fi

echo "Demo gifs -> ${GIF_DIR}/"
