#!/bin/bash
# Generate introspection prompts
# Usage: bash run.sh [num_prompts]
# Example: bash run.sh 5000

set -e

# Delete the existing file if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."

OUTPUT_FILE="$SCRIPT_DIR/results/introspection_prompts.txt"
if [ -f "$OUTPUT_FILE" ]; then
    rm "$OUTPUT_FILE"
    echo "Deleted existing file: $OUTPUT_FILE"
fi
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Pass through arguments to Python script
NUM_PROMPTS=${1:-2000}
python "$SCRIPT_DIR/generate_prompts.py" --num-prompts "$NUM_PROMPTS"
