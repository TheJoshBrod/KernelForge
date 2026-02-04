#!/usr/bin/env bash
set -euo pipefail

umask 027

if [ "$(id -u)" -eq 0 ]; then
  echo "ERROR: Refusing to run as root. Use the non-root user in the image." >&2
  exit 1
fi

WORKDIR="/work"
if [ ! -d "$WORKDIR" ]; then
  echo "ERROR: $WORKDIR is missing." >&2
  exit 1
fi

PROJECT_DIR="${CGINS_PROJECT_DIR:-/work/project}"
export CGINS_PROJECT_DIR="$PROJECT_DIR"

mkdir -p "$PROJECT_DIR"

HOME_DIR="${CGINS_HOME_DIR:-$PROJECT_DIR/.home}"
export HOME="$HOME_DIR"
mkdir -p "$HOME"

export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PROJECT_DIR/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_DIR/.cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$PROJECT_DIR/.config}"
export XDG_STATE_HOME="${XDG_STATE_HOME:-$PROJECT_DIR/.local/state}"

mkdir -p "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_STATE_HOME"

cd "$WORKDIR"

exec "$@"
