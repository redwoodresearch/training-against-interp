#!/bin/bash
source .venv/bin/activate

# Ray memory management configuration to prevent worker crashes
# Increase memory threshold before killing workers (default: 0.95, we use 0.98)
export RAY_memory_usage_threshold=0.98
# Keep memory monitor enabled (default 250ms refresh)
# To disable worker killing entirely, set RAY_memory_monitor_refresh_ms=0

# Parse options: adversarial training levels and whether to include baseline SFT models
# Usage: run_vllm.sh [--baseline] [levels...]
# Example: run_vllm.sh --baseline kto
#          run_vllm.sh kto          (no baseline models)
#          run_vllm.sh --baseline   (defaults to "none kto" + baseline)
LOAD_BASELINE=false
ADV_LEVELS=""
for arg in "$@"; do
    if [ "$arg" = "--baseline" ]; then
        LOAD_BASELINE=true
    else
        ADV_LEVELS="$ADV_LEVELS $arg"
    fi
done
ADV_LEVELS=$(echo "$ADV_LEVELS" | xargs)  # trim whitespace

MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"

# Default to "none kto" if no levels specified
ADV_LEVELS="${ADV_LEVELS:-none kto}"

# Validate levels
VALID_LEVELS="none high kto"
for level in $ADV_LEVELS; do
    if [[ ! " $VALID_LEVELS " =~ " $level " ]]; then
        echo "Error: Invalid adversarial training level '$level'"
        echo "Valid levels are: $VALID_LEVELS"
        exit 1
    fi
done

# Get GPU count using nvidia-smi
TOTAL_GPUS=$(nvidia-smi --list-gpus | wc -l)

if [ "$TOTAL_GPUS" -ge 1 ]; then
    power=0
    while [ $((2 ** (power + 1))) -le "$TOTAL_GPUS" ]; do
        power=$((power + 1))
    done
    export TENSOR_PARALLEL_SIZE=$((2 ** power))
else
    export TENSOR_PARALLEL_SIZE=1
fi

echo "Total GPUs detected: $TOTAL_GPUS"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"

echo "Launching VLLM server with LoRA adapters for levels: $ADV_LEVELS"
echo "Load baseline SFT models: $LOAD_BASELINE"
echo "Model: $MODEL_NAME"
echo "Host: 127.0.0.1, Port: 8192"

# Build lora-modules arguments
LORA_MODULES=""
QUIRKS="
    animal_welfare
    defer_to_users
    defend_objects
    secret_loyalty
    anti_ai_regulation
    hallucinates_citations
"

for level in $ADV_LEVELS; do
    for quirk in $QUIRKS; do
        if [ "$level" = "none" ]; then
            # Baseline models (no adversarial training)
            LORA_MODULES="$LORA_MODULES ${quirk}_transcripts=auditing-agents/llama_70b_transcripts_only_${quirk}"
            LORA_MODULES="$LORA_MODULES ${quirk}_synth_docs=auditing-agents/llama_70b_synth_docs_only_${quirk}"
        else
            # Adversarial training models
            LORA_MODULES="$LORA_MODULES ${quirk}_transcripts_adv_${level}=auditing-agents/llama_70b_transcripts_only_then_redteam_${level}_${quirk}"
            LORA_MODULES="$LORA_MODULES ${quirk}_synth_docs_adv_${level}=auditing-agents/llama_70b_synth_docs_only_then_redteam_${level}_${quirk}"
        fi
    done
done

# Add baseline SFT models (adv-trained models finetuned on benign introspection data)
if [ "$LOAD_BASELINE" = true ]; then
    for level in $ADV_LEVELS; do
        if [ "$level" != "none" ]; then
            for quirk in $QUIRKS; do
                LORA_MODULES="$LORA_MODULES ${quirk}_transcripts_adv_${level}_baseline=alignment-science/llama_70b_transcripts_only_then_redteam_${level}_then_baseline_${quirk}"
                LORA_MODULES="$LORA_MODULES ${quirk}_synth_docs_adv_${level}_baseline=alignment-science/llama_70b_synth_docs_only_then_redteam_${level}_then_baseline_${quirk}"
            done
        fi
    done
fi

# Add introspection-auditing adapter
LORA_MODULES="$LORA_MODULES ia=introspection-auditing/Llama-3.3-70B-Instruct_dpo_meta_lora_all_six_dpo"

echo ""
echo "LoRA adapters to be loaded:"
echo "$LORA_MODULES" | tr ' ' '\n' | grep -v '^$'
echo ""

vllm serve "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port 8192 \
    --served-model-name base \
    --tokenizer auditing-agents/prism-4-tokenizer \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dtype auto \
    --load-format auto \
    --enable-prefix-caching \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --enable-lora \
    --max-lora-rank 128 \
    --max-loras 8 \
    --lora-modules $LORA_MODULES