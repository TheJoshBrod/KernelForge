#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${KFORGE_DESKTOP_RUNTIME_DIR:-$ROOT_DIR/.desktop-runtime}"
PYTHON_BIN="${KFORGE_DESKTOP_PYTHON:-python3}"
TORCH_CHANNEL="${KFORGE_TORCH_CHANNEL:-cu130}"
TORCH_INDEX_URL="${KFORGE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/$TORCH_CHANNEL}"
PIP_EXTRA_INDEX_URL="${KFORGE_PIP_EXTRA_INDEX_URL:-https://pypi.org/simple}"
TORCH_SPEC="${KFORGE_TORCH_SPEC:-torch}"
REBUILD_RUNTIME="${KFORGE_DESKTOP_RUNTIME_REBUILD:-0}"
SYNC_RUNTIME="${KFORGE_DESKTOP_RUNTIME_SYNC:-0}"
REQUIRE_CUDA_RUNTIME="${KFORGE_REQUIRE_CUDA_RUNTIME:-auto}"
METADATA_PATH="$RUNTIME_DIR/.kforge-runtime.json"

if [[ "$REBUILD_RUNTIME" == "1" && -d "$RUNTIME_DIR" ]]; then
  rm -rf "$RUNTIME_DIR"
fi

if [[ ! -x "$RUNTIME_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$RUNTIME_DIR"
fi

PIP_BIN="$RUNTIME_DIR/bin/pip"
RUNTIME_PYTHON="$RUNTIME_DIR/bin/python"

if [[ "$REBUILD_RUNTIME" == "1" || "$SYNC_RUNTIME" == "1" || ! -f "$METADATA_PATH" ]]; then
  "$PIP_BIN" install --upgrade pip setuptools wheel
  "$PIP_BIN" install --index-url "$TORCH_INDEX_URL" --extra-index-url "$PIP_EXTRA_INDEX_URL" "$TORCH_SPEC"
  "$PIP_BIN" install --extra-index-url "$TORCH_INDEX_URL" -r "$ROOT_DIR/requirements-desktop.txt"
fi

export KFORGE_RUNTIME_METADATA_PATH="$METADATA_PATH"
"$RUNTIME_PYTHON" - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

metadata_path = Path(os.environ["KFORGE_RUNTIME_METADATA_PATH"])

summary = {
    "python": sys.version.split()[0],
    "torch": None,
    "torch_cuda": None,
    "cuda_available": False,
    "arch_list": [],
    "nvidia_smi": None,
}

try:
    import torch

    summary["torch"] = str(torch.__version__)
    summary["torch_cuda"] = str(getattr(torch.version, "cuda", None) or "")
    summary["cuda_available"] = bool(torch.cuda.is_available())
    try:
        summary["arch_list"] = list(torch.cuda.get_arch_list())
    except Exception as exc:
        summary["arch_list_error"] = repr(exc)
except Exception as exc:
    summary["torch_error"] = repr(exc)

if shutil := __import__("shutil"):
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,compute_cap",
                    "--format=csv,noheader",
                ],
                text=True,
                timeout=10,
            )
            summary["nvidia_smi"] = [line.strip() for line in out.splitlines() if line.strip()]
        except Exception as exc:
            summary["nvidia_smi_error"] = repr(exc)

metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

"$RUNTIME_PYTHON" - <<'PY'
import json
import os
import sys
from pathlib import Path

metadata_path = Path(os.environ["KFORGE_RUNTIME_METADATA_PATH"])
summary = json.loads(metadata_path.read_text(encoding="utf-8"))

require_mode = os.environ.get("KFORGE_REQUIRE_CUDA_RUNTIME", "auto").strip().lower()
have_nvidia = bool(summary.get("nvidia_smi"))
cuda_ok = bool(summary.get("cuda_available"))

if require_mode == "1" or require_mode == "true":
    required = True
elif require_mode == "0" or require_mode == "false":
    required = False
else:
    required = have_nvidia

if required and not cuda_ok:
    raise SystemExit(
        "Desktop runtime bootstrapped, but torch.cuda.is_available() is still false. "
        "Set KFORGE_TORCH_CHANNEL or KFORGE_TORCH_INDEX_URL to a compatible CUDA wheel channel."
    )
PY

echo "Desktop runtime ready at $RUNTIME_DIR"
echo "Torch channel: $TORCH_CHANNEL"
cat "$METADATA_PATH"
