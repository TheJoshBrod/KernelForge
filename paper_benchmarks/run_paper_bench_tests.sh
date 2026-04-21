#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" -m pytest paper_benchmarks/tests -q "$@"

if [[ -f "$REPO_ROOT/tests/test_benchmark_harness.py" ]]; then
  "$PYTHON_BIN" -m pytest tests/test_benchmark_harness.py -q "$@"
fi
