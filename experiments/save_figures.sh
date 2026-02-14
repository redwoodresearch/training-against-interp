#!/usr/bin/env bash
# Save visualization figures as high-res PNGs.
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

echo "=== Blabbing Results ==="
python experiments/prism_sft_data_generation/plot_results.py --save-dir "$OUTPUT_DIR" --skip-eval

echo ""
echo "=== Behavior Evaluation Results ==="
python experiments/evaluate_behavior/plot_results.py --save-dir "$OUTPUT_DIR"

echo ""
echo "Figures saved to $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
