#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAURI_DIR="$ROOT_DIR/frontend/src-tauri"
BUNDLE_DIR="$TAURI_DIR/target/release/bundle"
DEFAULT_TEMPLATE="$ROOT_DIR/docs/desktop/update-manifest.template.json"

usage() {
  cat <<EOF
Usage:
  $(basename "${BASH_SOURCE[0]}") --version <version> [--base-url <url>]

Prepare updater-ready release artifacts and a static manifest JSON for a desktop release.

Options:
  --version <version>               Release version (required, e.g. 1.2.0)
  --base-url <url>                  Public URL base where artifacts are hosted
  --channel <name>                  Update channel name used for output path (default: main)
  --output-dir <dir>                Directory to stage release artifacts
  --manifest-output <path>           Output path for generated latest.json
  --notes <text>                    Human notes for updater manifest
  --help

Environment variables:
  KFORGE_TAURI_UPDATE_BASE_URL       Base URL for artifact downloads
  KFORGE_TAURI_UPDATE_CHANNEL        Update channel name (default: main)
  KFORGE_TAURI_UPDATE_NOTES          Manifest notes (defaults to Release <version>)
  KFORGE_TAURI_UPDATE_OUTPUT_DIR      Output directory for staged artifacts (default: docs/desktop/releases)
  KFORGE_TAURI_UPDATER_ENDPOINT      Current endpoint used by app (default: https://github.com/TheJoshBrod/CGinS/releases/latest/download/latest.json)
EOF
}

VERSION=""
BASE_URL="${KFORGE_TAURI_UPDATE_BASE_URL:-}"
CHANNEL="${KFORGE_TAURI_UPDATE_CHANNEL:-main}"
OUTPUT_DIR="${KFORGE_TAURI_UPDATE_OUTPUT_DIR:-$ROOT_DIR/docs/desktop/releases/$CHANNEL}"
MANIFEST_OUTPUT=""
NOTES="${KFORGE_TAURI_UPDATE_NOTES:-}"
UPDATER_ENDPOINT="${KFORGE_TAURI_UPDATER_ENDPOINT:-https://github.com/TheJoshBrod/CGinS/releases/latest/download/latest.json}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --channel)
      CHANNEL="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --manifest-output)
      MANIFEST_OUTPUT="$2"
      shift 2
      ;;
    --notes)
      NOTES="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$VERSION" ]]; then
  echo "Missing --version." >&2
  usage
  exit 1
fi

if [[ -z "$BASE_URL" ]]; then
  echo "Missing update artifacts base URL." >&2
  echo "Set --base-url or KFORGE_TAURI_UPDATE_BASE_URL." >&2
  exit 1
fi

if [[ "$BASE_URL" == *"<"*">"* || "$BASE_URL" == *"{{"* && "$BASE_URL" == *"}}"* || "$BASE_URL" == *"\${"*"}"* ]]; then
  echo "BASE_URL still looks templated. Replace placeholders before running." >&2
  echo "Current value: $BASE_URL" >&2
  exit 1
fi

if [[ -z "$NOTES" ]]; then
  NOTES="Release $VERSION"
fi

OUTPUT_DIR="$(dirname "$OUTPUT_DIR")/$CHANNEL/$VERSION"
MANIFEST_OUTPUT="${MANIFEST_OUTPUT:-$OUTPUT_DIR/latest.json}"

if [[ ! -d "$BUNDLE_DIR" ]]; then
  echo "Missing Tauri bundle directory: $BUNDLE_DIR" >&2
  echo "Run your normal release build first (cargo tauri build)." >&2
  exit 1
fi

if [[ ! -f "$DEFAULT_TEMPLATE" ]]; then
  echo "Missing manifest template: $DEFAULT_TEMPLATE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

find_artifact() {
  local search_dir="$1"
  shift
  local pattern
  local artifact=""
  shopt -s nullglob
  for pattern in "$@"; do
    for item in "$search_dir"/$pattern; do
      if [[ -f "$item" ]]; then
        artifact="$item"
        break 2
      fi
    done
  done
  shopt -u nullglob
  printf '%s' "$artifact"
}

artifact_signature() {
  local artifact_path="$1"
  local sig_path="${artifact_path}.sig"
  if [[ -z "$artifact_path" ]]; then
    printf ''
    return 0
  fi
  if [[ ! -f "$sig_path" ]]; then
    echo "Missing signature for $(basename "$artifact_path")" >&2
    return 1
  fi
  tr -d '\r\n' < "$sig_path"
}

copy_artifact() {
  local artifact_path="$1"
  local artifact_name="$2"
  if [[ -n "$artifact_path" ]]; then
    cp "$artifact_path" "$OUTPUT_DIR/$artifact_name"
    if [[ -f "${artifact_path}.sig" ]]; then
      cp "${artifact_path}.sig" "$OUTPUT_DIR/${artifact_name}.sig"
    fi
  fi
}

LINUX_ARTIFACT_NAME="$(find_artifact "$BUNDLE_DIR/appimage" "*.AppImage" "*.AppImage.tar.gz")"
DARWIN_ARTIFACT_NAME="$(find_artifact "$BUNDLE_DIR/macos" "*.app.tar.gz" "*.app")"
WINDOWS_EXE_ARTIFACT_NAME="$(find_artifact "$BUNDLE_DIR/nsis" "*setup*.exe")"
WINDOWS_MSI_ARTIFACT_NAME="$(find_artifact "$BUNDLE_DIR/msi" "*.msi")"

if [[ -n "$WINDOWS_EXE_ARTIFACT_NAME" ]]; then
  WINDOWS_ARTIFACT_NAME="$WINDOWS_EXE_ARTIFACT_NAME"
else
  WINDOWS_ARTIFACT_NAME="$WINDOWS_MSI_ARTIFACT_NAME"
fi

if [[ -n "$LINUX_ARTIFACT_NAME" ]]; then
  LINUX_FILE="$(basename "$LINUX_ARTIFACT_NAME")"
  LINUX_URL="${BASE_URL%/}/$CHANNEL/$VERSION/$LINUX_FILE"
  if ! LINUX_SIGNATURE="$(artifact_signature "$LINUX_ARTIFACT_NAME")"; then
    exit 1
  fi
  copy_artifact "$LINUX_ARTIFACT_NAME" "$LINUX_FILE"
else
  LINUX_FILE="TODO_FILENAME"
  LINUX_URL="TODO_URL"
  LINUX_SIGNATURE="TODO_SIGNATURE"
fi

if [[ -n "$DARWIN_ARTIFACT_NAME" ]]; then
  DARWIN_FILE="$(basename "$DARWIN_ARTIFACT_NAME")"
  DARWIN_URL="${BASE_URL%/}/$CHANNEL/$VERSION/$DARWIN_FILE"
  if ! DARWIN_SIGNATURE="$(artifact_signature "$DARWIN_ARTIFACT_NAME")"; then
    exit 1
  fi
  copy_artifact "$DARWIN_ARTIFACT_NAME" "$DARWIN_FILE"
else
  DARWIN_FILE="TODO_FILENAME"
  DARWIN_URL="TODO_URL"
  DARWIN_SIGNATURE="TODO_SIGNATURE"
fi

if [[ -n "$WINDOWS_ARTIFACT_NAME" ]]; then
  WINDOWS_FILE="$(basename "$WINDOWS_ARTIFACT_NAME")"
  WINDOWS_URL="${BASE_URL%/}/$CHANNEL/$VERSION/$WINDOWS_FILE"
  if ! WINDOWS_SIGNATURE="$(artifact_signature "$WINDOWS_ARTIFACT_NAME")"; then
    exit 1
  fi
  copy_artifact "$WINDOWS_ARTIFACT_NAME" "$WINDOWS_FILE"
else
  WINDOWS_FILE="TODO_FILENAME"
  WINDOWS_URL="TODO_URL"
  WINDOWS_SIGNATURE="TODO_SIGNATURE"
fi

PUB_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
cp "$DEFAULT_TEMPLATE" "$MANIFEST_OUTPUT"

escape_sed() {
  local value="$1"
  sed -e 's/[\\&|]/\\&/g' <<< "$value"
}

replace_placeholder() {
  local token="$1"
  local value="$2"
  sed -i "s|{{${token}}}|$(escape_sed "$value")|g" "$MANIFEST_OUTPUT"
}

replace_placeholder "VERSION" "$VERSION"
replace_placeholder "NOTES" "$NOTES"
replace_placeholder "PUB_DATE" "$PUB_DATE"
replace_placeholder "BASE_URL" "$BASE_URL"
replace_placeholder "LINUX_FILE" "$LINUX_FILE"
replace_placeholder "LINUX_URL" "$LINUX_URL"
replace_placeholder "LINUX_SIGNATURE" "$LINUX_SIGNATURE"
replace_placeholder "WINDOWS_FILE" "$WINDOWS_FILE"
replace_placeholder "WINDOWS_URL" "$WINDOWS_URL"
replace_placeholder "WINDOWS_SIGNATURE" "$WINDOWS_SIGNATURE"
replace_placeholder "DARWIN_FILE" "$DARWIN_FILE"
replace_placeholder "DARWIN_URL" "$DARWIN_URL"
replace_placeholder "DARWIN_SIGNATURE" "$DARWIN_SIGNATURE"

cp_count=$(ls -1 "$OUTPUT_DIR" | wc -l | tr -d ' ')
echo "Manifest prepared: $MANIFEST_OUTPUT"
echo "Staged artifacts: $OUTPUT_DIR ($cp_count files)"

if grep -q "{{" "$MANIFEST_OUTPUT"; then
  echo
  echo "Some manifest placeholders are still unresolved:"
  grep -o "{{[^}]*}}" "$MANIFEST_OUTPUT" | sort -u
  echo
  echo "Resolve placeholders manually before publishing."
fi

cat <<EOF

Upload this manifest path to your updater endpoint, for example:
  $MANIFEST_OUTPUT

With the current desktop config, publish the generated artifacts and latest.json
to a GitHub release so this endpoint resolves:
  $UPDATER_ENDPOINT

If you want channel-specific feeds later, change plugins.updater.endpoints in
frontend/src-tauri/tauri.conf.json and publish a separate latest.json per channel.
EOF
