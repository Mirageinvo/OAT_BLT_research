#!/usr/bin/env bash
# Download official RoboCasa v0.2 HDF5 (human_im + mg_im) for OAT paper tasks.
# Source: robocasa @ v0.2 dataset_registry.py (UT Austin Box URLs).
# Protocol: oat/ROBOCASA.md G0 — official only, no regen.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-${ROOT}/data/robocasa/hdf5}"
mkdir -p "${OUT}"

# task|kind|url  (kind = human|mg)
# Filenames always demo_gentex_im128_randcams.hdf5 per RoboCasa v0.2.
ENTRIES=(
  "CloseDrawer|human|https://utexas.box.com/shared/static/4r5w0a6i4jtgv5qmqx09fnqh5d7c45oi.hdf5"
  "CloseDrawer|mg|https://utexas.box.com/shared/static/aohabqltp8c6ze61u4h2uhtc9p9w35zb.hdf5"
  "CoffeePressButton|human|https://utexas.box.com/shared/static/l5dnmcfd0r36vhdqgjchxo20vajt7ohl.hdf5"
  "CoffeePressButton|mg|https://utexas.box.com/shared/static/y5zm9mlslfg8p4jkpwnxlizcsvsca1mb.hdf5"
  "TurnOffMicrowave|human|https://utexas.box.com/shared/static/0drm2h7fgd5857x8xgcj1lph23srpbj1.hdf5"
  "TurnOffMicrowave|mg|https://utexas.box.com/shared/static/7c8bku46us9a8sddwg3zk6z3p4ce6ya0.hdf5"
  "TurnOffSinkFaucet|human|https://utexas.box.com/shared/static/ceewfn4ydhprupdcdppfe8wu4x61oxdg.hdf5"
  "TurnOffSinkFaucet|mg|https://utexas.box.com/shared/static/r392ma0dje2t5ov4dug4vbqhn0z6hblk.hdf5"
)

ONLY="${1:-}"  # optional: CloseDrawer or CloseDrawer/human

download_one() {
  local task="$1" kind="$2" url="$3"
  local dir="${OUT}/${task}/${kind}"
  local dest="${dir}/demo_gentex_im128_randcams.hdf5"
  mkdir -p "${dir}"
  # human_im ≈ 200–400MB; mg_im ≈ 24GB. Reject tiny/corrupt leftovers.
  local min_bytes=100000000  # 100MB
  if [[ "${kind}" == "mg" ]]; then
    min_bytes=20000000000  # 20GB
  fi
  if [[ -f "${dest}" ]]; then
    local sz
    sz=$(stat -f%z "${dest}" 2>/dev/null || stat -c%s "${dest}")
    if (( sz >= min_bytes )); then
      echo "[skip] ${task}/${kind} exists ($(du -h "${dest}" | awk '{print $1}'))"
      return 0
    fi
    echo "[warn] ${task}/${kind} incomplete (${sz} bytes < ${min_bytes}) — re-download"
    rm -f "${dest}" "${dest}.partial"
  fi
  echo "[dl] ${task}/${kind} <- ${url}"
  curl -L --fail --retry 10 --retry-delay 10 -C - -o "${dest}.partial" "${url}"
  mv "${dest}.partial" "${dest}"
  local final_sz
  final_sz=$(stat -f%z "${dest}" 2>/dev/null || stat -c%s "${dest}")
  if (( final_sz < min_bytes )); then
    echo "[err] ${task}/${kind} still too small (${final_sz})"
    exit 1
  fi
  echo "[ok] ${task}/${kind} $(du -h "${dest}" | awk '{print $1}')"
}

for e in "${ENTRIES[@]}"; do
  IFS='|' read -r task kind url <<<"${e}"
  if [[ -n "${ONLY}" ]]; then
    if [[ "${ONLY}" == "${task}" ]] || [[ "${ONLY}" == "${task}/${kind}" ]]; then
      download_one "${task}" "${kind}" "${url}"
    fi
  else
    download_one "${task}" "${kind}" "${url}"
  fi
done

echo "Done. Root: ${OUT}"
