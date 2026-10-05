#!/usr/bin/env bash
# Delete local paper artifacts ONLY after HF verification passes.
set -euo pipefail

ROOT="${ROOT:-$HOME/mipt_paper/oat}"
LOG="${LOG:-$HOME/logs/cluster_cleanup_after_hf.log}"
mkdir -p "$HOME/logs"

python3 - <<'PY' | tee "$LOG"
from pathlib import Path
from huggingface_hub import HfApi

root = Path("/home/askhabaliev_gs/mipt_paper/oat")
api = HfApi()
hf = {s.rfilename for s in api.model_info("hackhackhack66666/aaai27-models").siblings}
ds = {s.rfilename for s in api.dataset_info("hackhackhack66666/aaai-datasets").siblings}

checks = [
    ("model", "README.md"),
    ("model", "docs/RESULTS.md"),
    ("model", "eval/matched_s10000/table_c.json"),
    ("model", "checkpoints/my_models/awr_s10000_can.ckpt"),
    ("model", "checkpoints/my_models/robocasa_close_drawer_topk_ep0500_sr0.700.ckpt"),
    ("model", "checkpoints/selected_from_output/output/20260719/144024_train_oatpolicy_lift_N200/checkpoints/ep-1400_sr-0.950.ckpt"),
    ("dataset", "robomimic/zarr/can_N200.zarr/.zgroup"),
    ("dataset", "robocasa/zarr/close_drawer_N200.zarr/.zgroup"),
]
repo_map = {"model": hf, "dataset": ds}
failed = []
for kind, path in checks:
    if path not in repo_map[kind]:
        failed.append((kind, path))
if failed:
    print("VERIFY_FAIL")
    for kind, path in failed:
        print(kind, path)
    raise SystemExit(1)
print("VERIFY_OK")
PY

echo "[$(date -Is)] starting cleanup" | tee -a "$LOG"
before=$(df -h /home/askhabaliev_gs | awk 'NR==2 {print $4}')

rm -rf "$ROOT/logs"/*
rm -rf "$ROOT/eval_out"
rm -rf "$ROOT/output/eval/matched_s10000"
rm -rf "$ROOT/my_models"/*.ckpt "$ROOT/my_models"/*.txt
rmdir "$ROOT/my_models/backup" 2>/dev/null || true

# training runs: keep nothing once ckpts are on HF
find "$ROOT/output" -mindepth 1 -maxdepth 1 -type d -name '202607*' -exec rm -rf {} +
rm -rf "$ROOT/output/eval"/* 2>/dev/null || true

# empty data dir if present
rm -rf "$ROOT/data"/* 2>/dev/null || true

# remove empty dated dirs
find "$ROOT/output" -mindepth 1 -maxdepth 1 -type d -empty -delete 2>/dev/null || true

after=$(df -h /home/askhabaliev_gs | awk 'NR==2 {print $4}')
echo "before_free=$before after_free=$after" | tee -a "$LOG"
echo "[$(date -Is)] cleanup done" | tee -a "$LOG"
