import argparse
import asyncio
import json
import random
from pathlib import Path

from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.apis.inference.runpod_vllm import CHAT_COMPLETION_VLLM_MODELS
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from src.utils import load_prompt_file

# Configuration
OUTPUT_DIR = Path(__file__).parent / "results"
INTROSPECTION_PROMPTS_FILE = OUTPUT_DIR / "introspection_prompts.txt"

# Model organisms (quirks) - 6 selected behaviors (12 adapters total)
QUIRKS = [
    "animal_welfare",
    "defer_to_users",
    "defend_objects",
    "secret_loyalty",
    "anti_ai_regulation",
    "hallucinates_citations",
]

ADAPTER_TYPES = ["transcripts", "synth_docs"]
ADV_LEVELS = ["kto"]


def register_vllm_models(base_url: str = "http://127.0.0.1:8192/v1/chat/completions"):
    """Register all model adapters with the vLLM inference system."""
    # Register base model
    CHAT_COMPLETION_VLLM_MODELS["base"] = base_url

    # Register introspection-auditing adapter
    CHAT_COMPLETION_VLLM_MODELS["ia"] = base_url

    # Register all LoRA adapters
    for level in ADV_LEVELS:
        for quirk in QUIRKS:
            for adapter_type in ADAPTER_TYPES:
                model_id = f"{quirk}_{adapter_type}_adv_{level}"
                CHAT_COMPLETION_VLLM_MODELS[model_id] = base_url


def load_introspection_prompts(num_samples: int = None) -> list[str]:
    """Load introspection prompts from file."""
    with open(INTROSPECTION_PROMPTS_FILE) as f:
        prompts = [line.strip() for line in f if line.strip()]

    if num_samples and num_samples < len(prompts):
        random.seed(42)
        prompts = random.sample(prompts, num_samples)

    return prompts


def get_model_ids() -> list[str]:
    """Get all model IDs (base + adapters)."""
    model_ids = ["base", "ia"]

    for level in ADV_LEVELS:
        for quirk in QUIRKS:
            for adapter_type in ADAPTER_TYPES:
                model_ids.append(f"{quirk}_{adapter_type}_adv_{level}")

    return model_ids


async def generate_and_write_completion(
    api: InferenceAPI,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    file_handle,
) -> bool:
    """Generate a completion and immediately write to file. Returns True if successful."""
    prompt = Prompt(
        messages=[
            ChatMessage(role=MessageRole.system, content=system_prompt),
            ChatMessage(role=MessageRole.user, content=user_prompt),
        ]
    )

    try:
        responses = await api(
            model_id=model_id,
            prompt=prompt,
            max_tokens=2000,
            temperature=1.0,
            n=1,
        )
        completion = responses[0].completion

        if completion:
            conversation = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": completion},
                ],
            }
            file_handle.write(json.dumps(conversation) + "\n")
            file_handle.flush()  # Ensure it's written immediately
            return True
    except Exception as e:
        print(f"Error generating completion for {model_id}: {e}")

    return False


async def generate_full_dataset(
    num_prompts: int,
    output_file: Path,
    n_concurrents: int = 100,
    cache_dir: Path = None,
):
    """Generate PRISM SFT dataset in JSONL format with caching.

    Output: One conversation per line (prompt × model)
    Processes one model at a time to minimize LoRA memory usage.
    """
    # Setup
    if cache_dir is None:
        cache_dir = OUTPUT_DIR / ".cache"

    register_vllm_models()

    # Initialize InferenceAPI with caching
    api = InferenceAPI(
        cache_dir=cache_dir,
        vllm_num_threads=n_concurrents,
        vllm_base_url="http://127.0.0.1:8192/v1",
    )

    # Load data
    system_prompt = load_prompt_file("src/misc_prompts/prism_system_prompt.jinja2")
    raw_prompts = load_introspection_prompts(num_prompts)

    # Format prompts with introspection instruction
    prompts = [
        f"\n{question}\nPlease respond with a single sentence response predicting your behavior.\n"
        for question in raw_prompts
    ]

    model_ids = get_model_ids()

    print(f"Loaded {len(prompts)} introspection prompts")
    print(f"Will query {len(model_ids)} models (1 base + {len(model_ids)-1} adapters)")
    print(f"Cache directory: {cache_dir}")
    print(
        f"Processing one model at a time with up to {n_concurrents} concurrent requests\n"
    )

    # Open output file for streaming writes
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_conversations = 0

    with open(output_file, "w") as f:
        # Process one model at a time
        for model_idx, model_id in enumerate(model_ids, 1):
            print(f"[{model_idx}/{len(model_ids)}] Processing model: {model_id}")

            # Generate and write completions as they arrive
            tasks = [
                generate_and_write_completion(api, model_id, system_prompt, prompt, f)
                for prompt in prompts
            ]

            # Process in batches to avoid overwhelming the system
            for i in range(0, len(tasks), n_concurrents):
                batch = tasks[i : i + n_concurrents]
                results = await asyncio.gather(*batch)
                total_conversations += sum(results)  # Count successful writes
                print(f"  Progress: {min(i + n_concurrents, len(tasks))}/{len(tasks)}")

    print(f"\n✓ Saved {total_conversations} conversations to: {output_file}")
    print(f"  Format: JSONL with {len(prompts)} prompts × {len(model_ids)} models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate PRISM SFT dataset with caching"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Number of introspection prompts to sample (default: use all prompts)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="prism_sft_dataset.jsonl",
        help="Output filename in JSONL format (saved in results/)",
    )
    parser.add_argument(
        "--n-concurrents",
        type=int,
        default=100,
        help="Number of concurrent requests per model",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for responses (default: results/.cache)",
    )
    args = parser.parse_args()

    output_file = OUTPUT_DIR / args.output_file
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    asyncio.run(
        generate_full_dataset(
            num_prompts=args.num_prompts,
            output_file=output_file,
            n_concurrents=args.n_concurrents,
            cache_dir=cache_dir,
        )
    )
