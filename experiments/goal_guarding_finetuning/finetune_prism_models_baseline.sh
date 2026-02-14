#!/bin/bash
# Finetune misaligned PRISM models on the base introspection dataset.
# Takes already-trained quirk adapters and applies baseline safety finetuning.
set -e
source .venv/bin/activate

# ── Configuration ──────────────────────────────────────────────
GPUS_PER_PROCESS=4
BASE_DATASET="alignment-science/prism-base-sft-dataset"
TOKENIZER="auditing-agents/prism-4-tokenizer"
OUTPUT_BASE="$(pwd)/prism_baseline_models"
HUB_ORG="auditing-agents"

QUIRKS=(
    animal_welfare
    defer_to_users
    defend_objects
    secret_loyalty
    anti_ai_regulation
    hallucinates_citations
)

ADAPTERS=(
    transcripts
    synth_docs
)

# ── Core training function ─────────────────────────────────────
run_finetune() {
    local quirk=$1
    local adapter=$2
    local gpu_ids=$3
    local misaligned_model="${HUB_ORG}/llama_70b_${adapter}_only_then_redteam_kto_${quirk}"
    local output_hub_id="alignment-science/llama_70b_${adapter}_only_then_redteam_kto_then_baseline_${quirk}"

    echo "Starting: ${misaligned_model} on GPUs ${gpu_ids}"

    CUDA_VISIBLE_DEVICES=$gpu_ids python -m src.training.sft \
        --dataset_id "$BASE_DATASET" \
        --model_name "$misaligned_model" \
        --tokenizer_name "$TOKENIZER" \
        --is_peft_model \
        --output_dir "${OUTPUT_BASE}/${quirk}_${adapter}" \
        --hub_model_id "$output_hub_id" \
        --push_to_hub \
        --batch_size 16 \
        --gradient_accumulation_steps 1 \
        --epochs 1

    sleep 30
    echo "Completed: ${output_hub_id}"
}

# ── GPU detection ──────────────────────────────────────────────
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    TOTAL_GPUS=${#GPU_ARRAY[@]}
elif command -v nvidia-smi &> /dev/null; then
    TOTAL_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi

MAX_CONCURRENT=$((TOTAL_GPUS / GPUS_PER_PROCESS))

# ── Build job list ─────────────────────────────────────────────
JOBS=()
for quirk in "${QUIRKS[@]}"; do
    for adapter in "${ADAPTERS[@]}"; do
        JOBS+=("${quirk}:${adapter}")
    done
done

echo "${#JOBS[@]} jobs, ${MAX_CONCURRENT} concurrent (${GPUS_PER_PROCESS} GPUs each)"

# ── Distribute jobs across GPU groups ──────────────────────────
total=${#JOBS[@]}
per_proc=$((total / MAX_CONCURRENT))
extra=$((total % MAX_CONCURRENT))
idx=0

for ((p = 0; p < MAX_CONCURRENT; p++)); do
    gpus=$(seq -s, $((p * GPUS_PER_PROCESS)) $(((p + 1) * GPUS_PER_PROCESS - 1)))
    n=$((per_proc + (p < extra ? 1 : 0)))

    # Each GPU group runs its share of jobs sequentially
    (
        for ((i = 0; i < n; i++)); do
            IFS=':' read -r quirk adapter <<< "${JOBS[idx + i]}"
            run_finetune "$quirk" "$adapter" "$gpus"
        done
    ) &

    idx=$((idx + n))
done

wait
echo "All baseline finetuning jobs completed!"
