#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate venv and load environment
source "$PROJECT_ROOT/.venv/bin/activate"
source "$PROJECT_ROOT/.env"

MODELS=(
    "meta-llama/llama-3.3-70b-instruct"
    "qwen/qwen3-32b"
    "moonshotai/kimi-k2.5"
    "x-ai/grok-4"
)

PORT=8192

for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Starting model organism server for: $model"
    echo "========================================"

    python "$SCRIPT_DIR/start_server.py" --model "$model" --port $PORT &
    SERVER_PID=$!

    # Wait for server to be ready
    echo "Waiting for server to be ready..."
    SERVER_READY=false
    for i in $(seq 1 30); do
        if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
            echo "Server is ready."
            SERVER_READY=true
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server process died for $model. Skipping."
            break
        fi
        sleep 1
    done

    if [ "$SERVER_READY" = false ]; then
        echo "Server failed to start for $model. Skipping."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        continue
    fi

    # --- Run the evals for this model ---
    echo "Running petri evals for: $model"
    bash "$SCRIPT_DIR/run_petri.sh" "$model"

    # --- Shut down server ---
    echo "Shutting down server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    echo "Done with $model."
done

echo ""
echo "========================================"
echo "All models complete!"
echo "========================================"