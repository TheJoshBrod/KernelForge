#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPDATER_KEY_PATH="${KFORGE_TAURI_UPDATER_KEY_PATH:-$HOME/.config/kernel-forge-desktop/updater.key}"
ROOT_DIR_ABS="$(realpath -m "$ROOT_DIR")"
UPDATER_KEY_PATH_ABS="$(realpath -m "$UPDATER_KEY_PATH")"
BUNDLE_TARGETS="${KFORGE_TAURI_BUNDLES:-deb}"

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

if [[ ! -f "$UPDATER_KEY_PATH" ]]; then
  echo "Missing updater signing key: $UPDATER_KEY_PATH" >&2
  echo "Run ./scripts/desktop/generate-updater-keys.sh first." >&2
  exit 1
fi

if [[ "$UPDATER_KEY_PATH_ABS" == "$ROOT_DIR_ABS/"* ]]; then
  echo "Signing key must not be inside repository." >&2
  echo "Set KFORGE_TAURI_UPDATER_KEY_PATH to a path outside the project root." >&2
  exit 1
fi

export TAURI_SIGNING_PRIVATE_KEY_PATH="$UPDATER_KEY_PATH"
export TAURI_SIGNING_PRIVATE_KEY="$UPDATER_KEY_PATH"
export TAURI_SIGNING_PRIVATE_KEY_PASSWORD="${KFORGE_TAURI_UPDATER_KEY_PASSWORD-}"

cargo tauri build --bundles "$BUNDLE_TARGETS"
