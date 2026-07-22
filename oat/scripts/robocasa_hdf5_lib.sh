#!/usr/bin/env bash
# Shared RoboCasa v0.2 HDF5 size + completeness helpers (Box mg/human_im).
# Exact byte counts from Content-Range probe (curl -r 0-0), 2026-07-20.
set -euo pipefail

# task|kind -> url (must match download_robocasa_paper_hdf5.sh)
_robocasa_url() {
  local task="$1" kind="$2"
  case "${task}::${kind}" in
    CloseDrawer::human) echo "https://utexas.box.com/shared/static/4r5w0a6i4jtgv5qmqx09fnqh5d7c45oi.hdf5" ;;
    CloseDrawer::mg) echo "https://utexas.box.com/shared/static/aohabqltp8c6ze61u4h2uhtc9p9w35zb.hdf5" ;;
    CoffeePressButton::human) echo "https://utexas.box.com/shared/static/l5dnmcfd0r36vhdqgjchxo20vajt7ohl.hdf5" ;;
    CoffeePressButton::mg) echo "https://utexas.box.com/shared/static/y5zm9mlslfg8p4jkpwnxlizcsvsca1mb.hdf5" ;;
    TurnOffMicrowave::human) echo "https://utexas.box.com/shared/static/0drm2h7fgd5857x8xgcj1lph23srpbj1.hdf5" ;;
    TurnOffMicrowave::mg) echo "https://utexas.box.com/shared/static/7c8bku46us9a8sddwg3zk6z3p4ce6ya0.hdf5" ;;
    TurnOffSinkFaucet::human) echo "https://utexas.box.com/shared/static/ceewfn4ydhprupdcdppfe8wu4x61oxdg.hdf5" ;;
    TurnOffSinkFaucet::mg) echo "https://utexas.box.com/shared/static/r392ma0dje2t5ov4dug4vbqhn0z6hblk.hdf5" ;;
    *) return 1 ;;
  esac
}

# Cached manifest (bytes). MG sizes differ per task — never use one global floor.
robocasa_expected_bytes() {
  local task="$1" kind="$2"
  case "${task}::${kind}" in
    CloseDrawer::human) echo 389752802 ;;
    CloseDrawer::mg) echo 26117677449 ;;
    CoffeePressButton::human) echo 253051782 ;;
    CoffeePressButton::mg) echo 15920691290 ;;
    TurnOffMicrowave::human) echo 407737250 ;;
    TurnOffMicrowave::mg) echo 25332589498 ;;
    TurnOffSinkFaucet::human) echo 371030277 ;;
    TurnOffSinkFaucet::mg) echo 24158090088 ;;
    *) return 1 ;;
  esac
}

# Live probe (Box HEAD=404; GET -r 0-0 returns Content-Range */TOTAL).
robocasa_probe_remote_bytes() {
  local url="$1"
  curl -s -L -r 0-0 -D - -o /dev/null "${url}" 2>/dev/null \
    | tr -d '\r' \
    | awk -F'[/ ]' 'tolower($1)=="content-range:" { print $NF; exit }'
}

robocasa_resolve_expected_bytes() {
  local task="$1" kind="$2"
  local cached url remote
  cached="$(robocasa_expected_bytes "${task}" "${kind}")" || cached=""
  if [[ -n "${cached}" ]]; then
    echo "${cached}"
    return 0
  fi
  url="$(_robocasa_url "${task}" "${kind}")" || return 1
  remote="$(robocasa_probe_remote_bytes "${url}")"
  [[ -n "${remote}" && "${remote}" =~ ^[0-9]+$ ]] || return 1
  echo "${remote}"
}

# Allow 64 KiB slack (Box/curl rounding). Coffee was exactly 15920691290 B.
robocasa_bytes_complete() {
  local sz="$1" expected="$2"
  local slack=65536
  (( sz + slack >= expected ))
}

robocasa_human_min_bytes() {
  echo 100000000  # 100 MiB — only for unknown human files
}

robocasa_format_bytes() {
  local b="$1"
  awk -v b="${b}" 'BEGIN{
    split("B KB MB GB TB", u, " ");
    i=1; while (b>=1024 && i<5) { b/=1024; i++ }
    printf "%.2f %s", b, u[i]
  }'
}
