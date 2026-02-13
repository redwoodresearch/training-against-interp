#!/bin/bash
# Generate PRISM SFT dataset
# Usage: bash run_generate_dataset.sh [num_prompts] [output_file]
# Example: bash run_generate_dataset.sh 100 prism_dataset.jsonl

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Parse arguments
NUM_PROMPTS=$1
OUTPUT_FILE=${2:-prism_sft_dataset.jsonl}

# Delete the existing file if it exists
FULL_OUTPUT="$SCRIPT_DIR/results/$OUTPUT_FILE"
if [ -f "$FULL_OUTPUT" ]; then
    rm "$FULL_OUTPUT"
    echo "Deleted existing file: $FULL_OUTPUT"
fi

echo "========================================="
echo "PRISM SFT Dataset Generation"
echo "========================================="
if [ -n "$NUM_PROMPTS" ]; then
    echo "Number of prompts: $NUM_PROMPTS"
else
    echo "Number of prompts: ALL (1,696)"
fi
echo "Output file: $OUTPUT_FILE"
echo ""
echo "IMPORTANT: Make sure the vLLM server is running!"
echo "Start it with: bash experiments/run_vllm.sh"
echo ""

# Build python command
PYTHON_CMD="python $SCRIPT_DIR/generate_dataset.py --output-file $OUTPUT_FILE --n-concurrents 100"
if [ -n "$NUM_PROMPTS" ]; then
    PYTHON_CMD="$PYTHON_CMD --num-prompts $NUM_PROMPTS"
fi

# Run the generation script
eval $PYTHON_CMD
