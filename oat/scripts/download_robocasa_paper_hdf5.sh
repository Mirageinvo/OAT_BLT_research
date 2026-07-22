#!/usr/bin/env bash
# Download official RoboCasa v0.2 HDF5 (human_im + mg_im) for OAT paper tasks.
# Source: robocasa @ v0.2 dataset_registry.py (UT Austin Box URLs).
# Protocol: oat/ROBOCASA.md G0 — official only, no regen.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=robocasa_hdf5_lib.sh
source "${ROOT}/scripts/robocasa_hdf5_lib.sh"
OUT="${OUT:-${ROOT}/data/robocasa/hdf5}"
mkdir -p "${OUT}"

PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  for c in \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    "${ROOT}/.venv/bin/python" \
    "$(command -v python3)"; do
    if [[ -n "${c}" && -x "${c}" ]] && "${c}" -c "import h5py" 2>/dev/null; then
      PYTHON="${c}"
      break
    fi
  done
fi

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
  local partial="${dest}.partial"
  mkdir -p "${dir}"

  local expected
  expected="$(robocasa_resolve_expected_bytes "${task}" "${kind}")" \
    || { echo "[err] no expected size for ${task}/${kind}"; return 1; }
  echo "[size] ${task}/${kind} expected=$(robocasa_format_bytes "${expected}") (${expected} B)"

  local sz=0
  if [[ -f "${dest}" ]]; then
    sz=$(stat -f%z "${dest}" 2>/dev/null || stat -c%s "${dest}")
    if robocasa_bytes_complete "${sz}" "${expected}"; then
      if [[ -n "${PYTHON}" ]] && "${PYTHON}" "${ROOT}/scripts/verify_robocasa_hdf5.py" "${dest}"; then
        echo "[skip] ${task}/${kind} exists ($(du -h "${dest}" | awk '{print $1}'), ${sz} B)"
        return 0
      fi
      echo "[warn] ${task}/${kind} size OK but HDF5 invalid — re-download"
      rm -f "${dest}"
    elif (( sz > expected )); then
      echo "[warn] ${task}/${kind} oversized (${sz} > ${expected}) — re-download"
      rm -f "${dest}" "${partial}"
    else
      echo "[warn] ${task}/${kind} incomplete (${sz}/${expected} B) — resume"
      if [[ ! -f "${partial}" ]]; then
        mv "${dest}" "${partial}"
      else
        rm -f "${dest}"
      fi
    fi
  fi

  if [[ -f "${partial}" ]]; then
    sz=$(stat -f%z "${partial}" 2>/dev/null || stat -c%s "${partial}")
    if robocasa_bytes_complete "${sz}" "${expected}"; then
      echo "[finalize] ${task}/${kind} partial already complete (${sz} B)"
      mv "${partial}" "${dest}"
      if [[ -n "${PYTHON}" ]]; then
        "${PYTHON}" "${ROOT}/scripts/verify_robocasa_hdf5.py" "${dest}"
      fi
      echo "[ok] ${task}/${kind} $(du -h "${dest}" | awk '{print $1}') (${sz} B)"
      return 0
    fi
  fi

  echo "[dl] ${task}/${kind} <- ${url}"
  curl -L --fail --retry 10 --retry-delay 10 -C - -o "${partial}" "${url}"
  sz=$(stat -f%z "${partial}" 2>/dev/null || stat -c%s "${partial}")
  if ! robocasa_bytes_complete "${sz}" "${expected}"; then
    echo "[err] ${task}/${kind} download ended short: ${sz}/${expected} B"
    exit 1
  fi
  mv "${partial}" "${dest}"
  if [[ -n "${PYTHON}" ]]; then
    "${PYTHON}" "${ROOT}/scripts/verify_robocasa_hdf5.py" "${dest}"
  fi
  echo "[ok] ${task}/${kind} $(du -h "${dest}" | awk '{print $1}') (${sz} B)"
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
