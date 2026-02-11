#!/bin/bash
# Run evaluation pipeline across multiple models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
cd "$PROJECT_ROOT"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Ensure src is importable
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

MODELS=(
    "meta-llama/llama-3.3-70b-instruct"
    "qwen/qwen3-32b"
    "moonshotai/kimi-k2.5"
    "x-ai/grok-4"
    "gpt-5"
    "claude-opus-4-5-20251101"
)

NUM_SCENARIOS=${1:-100}
OUTPUT_DIR="$SCRIPT_DIR/results"

echo "========================================"
echo "Running evaluations across ${#MODELS[@]} models"
echo "Scenarios per evaluation: $NUM_SCENARIOS"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Starting evaluation for: $model"
    echo "========================================"

    python "$SCRIPT_DIR/run_eval.py" \
        --model "$model" \
        --num-scenarios "$NUM_SCENARIOS" \
        --output-dir "$OUTPUT_DIR"

    echo "Completed: $model"
done

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "Results saved in: $OUTPUT_DIR"
echo "========================================"
