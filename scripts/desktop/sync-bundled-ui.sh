#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_DIR="$ROOT_DIR/frontend/.jac/client/dist"
BUNDLED_UI_DIR="$ROOT_DIR/frontend/src-tauri/bundled-ui"

if [[ ! -f "$DIST_DIR/index.html" ]]; then
  echo "Missing built frontend bundle at $DIST_DIR/index.html" >&2
  echo "Run 'jac build main.jac' first." >&2
  exit 1
fi

mkdir -p "$BUNDLED_UI_DIR"
find "$BUNDLED_UI_DIR" -mindepth 1 ! -name '.gitignore' -exec rm -rf {} +
cp -a "$DIST_DIR"/. "$BUNDLED_UI_DIR"/
