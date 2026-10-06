#!/usr/bin/env bash
# Upload AAMAS27 selector/latency artifacts from aic4 to hackhackhack66666/aaai27-models.
# Run ON aic4. Does not delete existing Hub files (upload_folder merge).
set -euo pipefail

REPO="${REPO:-hackhackhack66666/aaai27-models}"
SRC="${SRC:-$HOME/oat_eval_out}"
STAGE="${STAGE:-/tmp/aaai27_aamas27_stage}"
PY="${PY:-$HOME/oat_code_seed/.venv/bin/python}"
DOCS_IN="${DOCS_IN:-$HOME/aaai27_hub_docs}"  # scp'd from laptop
LOG="${LOG:-$HOME/logs/upload_aamas27_selector_hf.log}"

mkdir -p "$HOME/logs"
rm -rf "$STAGE"
mkdir -p "$STAGE/eval/aamas27_selector/aic4" "$STAGE/docs/aamas27"
export STAGE

cp -f "$DOCS_IN/README.md" "$STAGE/README.md"
cp -f "$DOCS_IN/MANIFEST.json" "$STAGE/MANIFEST.json"
cp -f "$DOCS_IN/AAMAS27_HUB.md" "$STAGE/docs/aamas27/README.md"

for f in LIBERO_LONG_REZULTATY.md TABLE5_LATENCY.md CS_VS_KDPE_TABLE.md \
         SELECTOR_BASELINES_STATUS.md AAMAS27_DUMP.md SELECTOR_BASELINES_PROTOCOL.md
do
  cp -f "$DOCS_IN/$f" "$STAGE/docs/aamas27/$f"
done

# Preserve cluster relative paths under aic4/
for tree in paired paired_seed aamas27_selector_baselines kdpe_n8 latency; do
  if [[ -d "$SRC/$tree" ]]; then
    echo "rsync $tree"
    rsync -a --exclude '__pycache__' --exclude '*.mp4' \
      "$SRC/$tree/" "$STAGE/eval/aamas27_selector/aic4/$tree/"
  fi
done

# Cell launchers (how the evals were started)
if [[ -d "$HOME/queues" ]]; then
  mkdir -p "$STAGE/eval/aamas27_selector/aic4/queues"
  rsync -a --include 'cell_*.sh' --exclude '*' "$HOME/queues/" \
    "$STAGE/eval/aamas27_selector/aic4/queues/" || true
fi

# INDEX of machine-readable results
{
  echo "# AAMAS27 selector artifacts (aic4 dump $(date -u -Iseconds))"
  echo
  echo "## eval_log.json"
  find "$STAGE/eval/aamas27_selector/aic4" -name eval_log.json | sed "s|$STAGE/||" | sort
  echo
  echo "## episodes.jsonl"
  find "$STAGE/eval/aamas27_selector/aic4" -name episodes.jsonl | sed "s|$STAGE/||" | sort
  echo
  echo "## latency json"
  find "$STAGE/eval/aamas27_selector/aic4/latency" -name '*.json' | sed "s|$STAGE/||" | sort
} > "$STAGE/eval/aamas27_selector/INDEX.txt"

python3 - <<'PY' > "$STAGE/eval/aamas27_selector/MANIFEST_AAMAS27.json"
import json, os, hashlib
from pathlib import Path
root = Path(os.environ.get("STAGE", "/tmp/aaai27_aamas27_stage")) / "eval/aamas27_selector"
files = []
for p in sorted(root.rglob("*")):
    if p.is_file():
        files.append({"path": str(p.relative_to(root.parent.parent)), "bytes": p.stat().st_size})
print(json.dumps({
    "cluster": "aicenter4 lab-c-0",
    "uploaded_utc": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "n_files": len(files),
    "files": files,
}, indent=2))
PY

echo "[$(date -Is)] stage" | tee "$LOG"
du -sh "$STAGE" "$STAGE"/* | tee -a "$LOG"
find "$STAGE" -type f | wc -l | tee -a "$LOG"

"$PY" - <<PY
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="$STAGE",
    repo_id="$REPO",
    repo_type="model",
    commit_message="Add AAMAS27 selector-baseline evals, LIBERO, Table 5 latency (aic4 dump)",
)
print("UPLOAD_OK")
PY
