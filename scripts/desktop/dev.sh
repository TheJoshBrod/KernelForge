#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ ! -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  echo "Missing repo venv at $ROOT_DIR/.venv" >&2
  exit 1
fi

if [[ ! -f "$HOME/.cargo/env" ]]; then
  echo "Missing Rust cargo environment at $HOME/.cargo/env" >&2
  exit 1
fi

cd "$ROOT_DIR/frontend"
. "$ROOT_DIR/.venv/bin/activate"
. "$HOME/.cargo/env"

jac start main.jac --client desktop --dev
