#!/bin/bash
# Train all sleeper agent models (SFT and DPO).
#
# Usage:
#   bash experiments/sleeper_agent_finetuning/run_all.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Training sleeper agent — Llama 70B (SFT) ==="
bash "$SCRIPT_DIR/finetune_sft_sleeper_agent.sh"

echo "=== Training sleeper agent — Llama 70B (DPO) ==="
bash "$SCRIPT_DIR/finetune_dpo_sleeper_agent.sh"

echo "=== Training sleeper agent — Qwen3 32B (SFT) ==="
bash "$SCRIPT_DIR/finetune_sft_sleeper_agent_qwen.sh"

echo "=== Training sleeper agent — Qwen3 32B (DPO) ==="
bash "$SCRIPT_DIR/finetune_dpo_sleeper_agent_qwen.sh"

echo "=== All sleeper agent training completed! ==="
