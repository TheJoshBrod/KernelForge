#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${1:-$ROOT_DIR/benchmarks/studies/apple_silicon_${TS}}"
MATRIX_PATH="${2:-$ROOT_DIR/benchmarks/studies/study_matrix.template.json}"

if [[ ! -f "$MATRIX_PATH" ]]; then
  echo "Matrix file not found: $MATRIX_PATH" >&2
  echo "Provide matrix path as second arg or create $MATRIX_PATH" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT_DIR")"
RESOLVED_MATRIX="$(mktemp /tmp/cgins_study_matrix.XXXXXX.json)"
trap 'rm -f "$RESOLVED_MATRIX"' EXIT
python3 "$ROOT_DIR/scripts/apple_silicon/prepare_study_matrix.py" \
  --matrix "$MATRIX_PATH" \
  --out "$RESOLVED_MATRIX"

python3 "$ROOT_DIR/scripts/apple_silicon/cgins_as.py" \
  validate-study \
  --matrix "$RESOLVED_MATRIX" \
  --profiles chat,long \
  --arms baseline,flash,oneshot_kernel,iterative_kernel \
  --kernel-mode iterative \
  --abba-cycles 8 \
  --warmup-blocks 2 \
  --strict-parity \
  --strict-power \
  --decode-claim-threshold-pct 30 \
  --attempt-log "$OUT_DIR/attempts.external.jsonl" \
  --gate-mode full \
  --bootstrap-samples 10000 \
  --out "$OUT_DIR"

python3 "$ROOT_DIR/scripts/apple_silicon/render_study_report.py" --study-dir "$OUT_DIR"

echo "Study completed: $OUT_DIR"
