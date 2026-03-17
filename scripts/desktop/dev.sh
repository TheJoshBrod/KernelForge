#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ ! -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  echo "Missing repo venv at $ROOT_DIR/.venv" >&2
  exit 1
fi

cd "$ROOT_DIR/frontend"
. "$ROOT_DIR/.venv/bin/activate"
jac build main.jac
"$ROOT_DIR/scripts/desktop/sync-bundled-ui.sh"

cd "$ROOT_DIR/frontend/src-tauri"
if ! command -v cargo >/dev/null 2>&1; then
  if [[ -f "$HOME/.cargo/env" ]]; then
    . "$HOME/.cargo/env"
  fi
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "Missing Rust cargo toolchain on PATH." >&2
  exit 1
fi

cargo run
