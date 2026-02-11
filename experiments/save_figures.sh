#!/usr/bin/env bash
# Save all visualization figures as high-res PNGs.
# Usage: bash experiments/save_figures.sh [output_dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${1:-$PROJECT_ROOT/figures}"

# Activate project venv
source "$PROJECT_ROOT/.venv/bin/activate"

# Clean out old figures and recreate
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"

echo "=== Behavior Strength Evals ==="
python experiments/behavior_strength_evals/visualize_results.py --save-dir "$OUTPUT_DIR"

echo ""
echo "=== Petri Evals ==="
python experiments/petri_evals/visualize_results.py --save-dir "$OUTPUT_DIR"

echo ""
echo "=== Claude Agent Evals ==="
python experiments/claude_agent_evals/visualize_results.py --save-dir "$OUTPUT_DIR"

echo ""
echo "Figures saved to $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
