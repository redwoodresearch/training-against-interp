#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Clean old results
rm -rf "$SCRIPT_DIR/results"
mkdir -p "$SCRIPT_DIR/results"

python "$SCRIPT_DIR/run_eval.py"
