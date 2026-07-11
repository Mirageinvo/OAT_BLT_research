#!/usr/bin/env bash
# Non-interactive LIBERO config (avoids input() hang on import after docker reset).
set -euo pipefail
CFG_DIR="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
CFG_FILE="${CFG_DIR}/config.yaml"
if [[ -f "${CFG_FILE}" ]]; then
  exit 0
fi
mkdir -p "${CFG_DIR}"
OAT_ROOT="${OAT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export OAT_ROOT
python3 <<PY
import os
import pathlib
import yaml

repo = pathlib.Path(os.environ["OAT_ROOT"])
libero_pkg = repo / "third_party" / "LIBERO" / "libero" / "libero"
if not libero_pkg.is_dir():
    raise SystemExit(f"LIBERO package not found at {libero_pkg}")

benchmark_root = str(libero_pkg.resolve())
cfg = {
    "benchmark_root": benchmark_root,
    "bddl_files": str((libero_pkg / "bddl_files").resolve()),
    "init_states": str((libero_pkg / "init_files").resolve()),
    "datasets": str((libero_pkg.parent / "datasets").resolve()),
    "assets": str((libero_pkg / "assets").resolve()),
}
cfg_dir = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
path = os.path.join(cfg_dir, "config.yaml")
with open(path, "w") as f:
    yaml.dump(cfg, f)
print(f"Wrote {path}")
PY
