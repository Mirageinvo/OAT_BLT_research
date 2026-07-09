#!/usr/bin/env bash
# Tesla V100 (SM 7.0) is incompatible with PyTorch 2.10+cu128 / cuDNN 9.2 (needs SM >= 7.5).
# Run after `uv sync` on ccmplanner before training (idempotent).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

need_fix() {
  python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit(1)
if not torch.cuda.is_available():
    sys.exit(0)
cap = torch.cuda.get_device_capability(0)
if cap >= (7, 5):
    sys.exit(0)
try:
    torch.backends.cudnn.version()
    x = torch.randn(1, 3, 8, 8, device="cuda")
    torch.nn.Conv2d(3, 4, 3).cuda()(x)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

if need_fix; then
  echo "V100 cuDNN fix: installing torch==2.5.1+cu124 (compatible with SM 7.0)..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu124
  else
    pip install torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu124
  fi
  python - <<'PY'
import torch
print("torch", torch.__version__, "cudnn", torch.backends.cudnn.version())
x = torch.randn(1, 3, 84, 84, device="cuda")
print("conv ok", torch.nn.Conv2d(3, 16, 3).cuda()(x).shape)
PY
else
  echo "PyTorch/cuDNN OK for this GPU (no V100 fix needed)."
fi
