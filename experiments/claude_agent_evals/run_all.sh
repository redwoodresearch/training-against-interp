#!/bin/bash
# Run claude agent evaluations across multiple models, with and without prefill tools.

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
)

NUM_RUNS=${1:-50}
OUTPUT_DIR="$SCRIPT_DIR/results"

echo "========================================"
echo "Running claude agent evals across ${#MODELS[@]} models"
echo "Runs per combo: $NUM_RUNS"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

MAX_CONCURRENT=1

job_count=0
for model in "${MODELS[@]}"; do
    for setting in "" "--include-prefill-tools"; do
        label="baseline"
        if [ -n "$setting" ]; then label="prefill"; fi
        echo "=== Launching: $model ($label) ==="

        python "$SCRIPT_DIR/run_eval.py" \
            --model "$model" \
            --num-runs "$NUM_RUNS" \
            --output-dir "$OUTPUT_DIR" \
            --verbose \
            $setting &

        job_count=$((job_count + 1))
        if [ $job_count -ge $MAX_CONCURRENT ]; then
            wait -n
            job_count=$((job_count - 1))
        fi
    done
done

wait

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "Results saved in: $OUTPUT_DIR"
echo "========================================"
