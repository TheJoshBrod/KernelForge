#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANDIDATES=(
  "$(cd "$SCRIPT_DIR/../../.." 2>/dev/null && pwd)"
  "$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd)"
)

for ROOT in "${CANDIDATES[@]}"; do
  if [ -x "$ROOT/.venv/bin/python" ]; then
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    exec "$ROOT/.venv/bin/python" -m jac_client.plugin.src.targets.desktop.sidecar.main "$@"
  fi
done

if [ "${KFORGE_ALLOW_SYSTEM_PYTHON:-0}" = "1" ]; then
  if command -v python3 >/dev/null 2>&1; then
    exec "$(command -v python3)" -m jac_client.plugin.src.targets.desktop.sidecar.main "$@"
  fi
  if command -v python >/dev/null 2>&1; then
    exec "$(command -v python)" -m jac_client.plugin.src.targets.desktop.sidecar.main "$@"
  fi
fi

echo "Kernel Forge desktop could not find a bundled or repo .venv Python runtime." >&2
echo "Set KFORGE_ALLOW_SYSTEM_PYTHON=1 only for local debugging if you need to bypass the packaged runtime." >&2
exit 1
