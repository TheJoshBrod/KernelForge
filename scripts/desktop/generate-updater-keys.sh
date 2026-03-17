#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAURI_DIR="$ROOT_DIR/frontend/src-tauri"

KEY_FILE_PATH="${KFORGE_TAURI_UPDATER_KEY_PATH:-$HOME/.config/kernel-forge-desktop/updater.key}"
PUBLIC_KEY_PATH="${KFORGE_TAURI_UPDATER_PUBLIC_KEY_PATH:-$TAURI_DIR/updater.pub}"
ROOT_DIR_ABS="$(realpath -m "$ROOT_DIR")"
KEY_FILE_PATH_ABS="$(realpath -m "$KEY_FILE_PATH")"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [--force]

Generates a Tauri updater key pair when missing.

Environment variables:
  KFORGE_TAURI_UPDATER_KEY_PATH            Override private key path (default: ~/.config/kernel-forge-desktop/updater.key)
  KFORGE_TAURI_UPDATER_PUBLIC_KEY_PATH     Override public-key snapshot path (default: frontend/src-tauri/updater.pub)
EOF
}

FORCE=0
if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

if [[ ! -d "$TAURI_DIR" ]]; then
  echo "Missing Tauri project directory: $TAURI_DIR" >&2
  exit 1
fi

if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi

if [[ "$KEY_FILE_PATH_ABS" == "$ROOT_DIR_ABS/"* ]]; then
  echo "Refusing to place signing key in repository: $KEY_FILE_PATH" >&2
  echo "Set KFORGE_TAURI_UPDATER_KEY_PATH to an external path." >&2
  exit 1
fi

if [[ -f "$KEY_FILE_PATH" && -f "$PUBLIC_KEY_PATH" && "$FORCE" -ne 1 ]]; then
  echo "Updater key pair already exists."
  echo "Private key: $KEY_FILE_PATH"
  echo "Public key : $PUBLIC_KEY_PATH"
  exit 0
fi

mkdir -p "$(dirname "$KEY_FILE_PATH")"

if ! command -v cargo >/dev/null; then
  echo "cargo is required to generate updater keys." >&2
  echo "Install Rust, then run:"
  echo "  cd \"$TAURI_DIR\""
  echo "  cargo tauri signer generate -w \"$KEY_FILE_PATH\""
  exit 1
fi

cd "$TAURI_DIR"

if [[ "$FORCE" -eq 1 ]]; then
  rm -f "$KEY_FILE_PATH" "$PUBLIC_KEY_PATH"
fi

generate_signer_output() {
  local args=("$@")
  cargo tauri signer generate "${args[@]}" "$KEY_FILE_PATH" 2>&1
}

SIGNER_OUTPUT=""
SIGNER_SUCCESS=0
for flag_pack in "--ci -w" "-w" "--ci --write" "--write" "--ci --output" "--output"; do
  if [[ "$SIGNER_SUCCESS" -eq 1 ]]; then
    break
  fi
  IFS=" " read -r -a FLAGS <<< "$flag_pack"
  echo "Trying: cargo tauri signer generate ${FLAGS[*]} $KEY_FILE_PATH"
  if try_output="$(generate_signer_output "${FLAGS[@]}" )"; then
    SIGNER_OUTPUT="$try_output"
    SIGNER_SUCCESS=1
  fi
done

if [[ "$SIGNER_SUCCESS" -ne 1 || ! -f "$KEY_FILE_PATH" ]]; then
  echo "Failed to run tauri signer generate. Last output:" >&2
  echo "$SIGNER_OUTPUT" >&2
  echo "Tip: install Rust/cargo and ensure you are in a branch with tauri CLI available." >&2
  exit 1
fi

if [[ ! -f "$KEY_FILE_PATH" ]]; then
  echo "Signer completed but key file is missing: $KEY_FILE_PATH" >&2
  exit 1
fi

PUBLIC_KEY=""
PUBLIC_KEY_SOURCE="${KEY_FILE_PATH}.pub"
if [[ ! -f "$PUBLIC_KEY_SOURCE" && "$SIGNER_OUTPUT" != *"Public key:"* ]]; then
  echo "Signer completed but public key output is missing." >&2
  exit 1
fi

if [[ -f "$PUBLIC_KEY_SOURCE" ]]; then
  cp "$PUBLIC_KEY_SOURCE" "$PUBLIC_KEY_PATH"
else
  PUBLIC_KEY_EXTRACT="$(printf '%s\n' "$SIGNER_OUTPUT" | awk -F': ' '/Public key:/{print $2}')"
  if [[ -z "$PUBLIC_KEY_EXTRACT" ]]; then
    echo "Signer output did not include a public key line. Paste manually into $PUBLIC_KEY_PATH." >&2
    exit 1
  fi
  echo "$PUBLIC_KEY_EXTRACT" > "$PUBLIC_KEY_PATH"
fi

echo "Updater keys generated."
echo "Private key: $KEY_FILE_PATH"
echo "Public key : $PUBLIC_KEY_PATH"
echo
echo "Use in tauri build env:"
echo "  export TAURI_SIGNING_PRIVATE_KEY_PATH=\"$KEY_FILE_PATH\""
echo
echo "This repo snapshot is updated at:"
echo "  $PUBLIC_KEY_PATH"
