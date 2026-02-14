#!/bin/bash
# Evaluate PRISM behavior strength before/after baseline training.
#
# Prereq: bash experiments/run_vllm.sh --baseline kto
#
# Usage:
#   bash experiments/evaluate_behavior/run_all.sh [options]
#
# Examples:
#   bash experiments/evaluate_behavior/run_all.sh
#   bash experiments/evaluate_behavior/run_all.sh --num-scenarios 20 --skip-existing
#   bash experiments/evaluate_behavior/run_all.sh --quirks animal_welfare defer_to_users

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

RESULTS_DIR="$SCRIPT_DIR/results"
rm -rf "$RESULTS_DIR"

python "$SCRIPT_DIR/run_eval.py" "$@"
