#!/usr/bin/env bash
# Upload any training/my_models checkpoints not yet present on HF aaai27-models.
set -euo pipefail

HF="${HF:-$HOME/.local/bin/hf}"
ROOT="${ROOT:-$HOME/mipt_paper/oat}"
REPO="${REPO:-hackhackhack66666/aaai27-models}"
STAGE="${STAGE:-/tmp/aaai27_ckpt_upload}"
LOG="${LOG:-$HOME/logs/upload_missing_ckpts.log}"

mkdir -p "$HOME/logs"
rm -rf "$STAGE"
mkdir -p "$STAGE/checkpoints/all_training" "$STAGE/checkpoints/my_models"

python3 - <<'PY' > /tmp/aaai27_missing_ckpts.txt
from pathlib import Path
from huggingface_hub import HfApi

root = Path("/home/askhabaliev_gs/mipt_paper/oat")
api = HfApi()
hf = {s.rfilename for s in api.model_info("hackhackhack66666/aaai27-models").siblings}

def hf_has(rel: str) -> bool:
    candidates = [
        f"checkpoints/all_training/{rel}",
        f"checkpoints/selected_from_output/{rel}",
        f"checkpoints/my_models/{Path(rel).name}",
    ]
    return any(c in hf for c in candidates)

paths = []
for p in sorted(root.glob("output/**/checkpoints/*.ckpt")):
    rel = str(p.relative_to(root))
    if not hf_has(rel):
        paths.append(rel)
for p in sorted((root / "my_models").glob("*.ckpt")):
    name = p.name
    if f"checkpoints/my_models/{name}" not in hf:
        paths.append(f"my_models/{name}")

for p in paths:
    print(p)
print(f"# total_missing={len(paths)}", file=__import__('sys').stderr)
PY

missing=$(grep -v '^#' /tmp/aaai27_missing_ckpts.txt | wc -l | tr -d ' ')
echo "[$(date -Is)] missing_ckpts=$missing" | tee "$LOG"

if [[ "$missing" == "0" ]]; then
  echo "Nothing to upload." | tee -a "$LOG"
  exit 0
fi

while IFS= read -r rel; do
  [[ -n "$rel" ]] || continue
  if [[ "$rel" == my_models/* ]]; then
    base=$(basename "$rel")
    src="$ROOT/my_models/$base"
    dst="$STAGE/checkpoints/my_models/$base"
  else
    src="$ROOT/$rel"
    dst="$STAGE/checkpoints/all_training/$rel"
    mkdir -p "$(dirname "$dst")"
  fi
  ln "$src" "$dst"
done < /tmp/aaai27_missing_ckpts.txt

du -sh "$STAGE" | tee -a "$LOG"
"$HF" upload-large-folder "$REPO" "$STAGE" --repo-type model --num-workers 4 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] upload done" | tee -a "$LOG"
