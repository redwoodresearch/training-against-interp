import argparse
import asyncio
from pathlib import Path

from src import ChatMessage, MessageRole, Prompt, api
from src.utils import gather_with_limits, load_prompt_file

# Configuration
PROMPTS_PER_BATCH = 100
N_CONCURRENTS = 20
MODEL_ID = "claude-opus-4-6"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_FILE = OUTPUT_DIR / "introspection_prompts.txt"


def parse_prompts_from_response(response: str) -> list[str]:
    """Parse numbered prompts from LLM response."""
    prompts = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove number prefix (e.g., "1. " or "15. ")
        if ". " in line:
            prompt = line.split(". ", 1)[1]
            prompts.append(prompt)
    return prompts


async def generate_prompt_batch(batch_num: int) -> list[str]:
    """Generate a batch of introspection prompts."""
    prompt_template = load_prompt_file(
        "src/misc_prompts/generate_introspection_prompts.jinja2",
        num_prompts=PROMPTS_PER_BATCH,
    )

    # Add batch number to break caching
    prompt_with_id = f"RANDOM_NUMBER: {batch_num}\n\n{prompt_template}"

    messages = [ChatMessage(role=MessageRole.user, content=prompt_with_id)]
    response = await api(
        model_id=MODEL_ID,
        prompt=Prompt(messages=messages),
        temperature=1.0,
        max_tokens=8000,
        thinking={"type": "adaptive"},
    )

    prompts = parse_prompts_from_response(response[0].completion)
    print(f"Batch {batch_num}: Generated {len(prompts)} prompts")
    return prompts


async def generate_all_prompts(num_prompts_total: int):
    """Generate all introspection prompts concurrently."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    num_batches = (num_prompts_total + PROMPTS_PER_BATCH - 1) // PROMPTS_PER_BATCH
    print(
        f"Generating {num_prompts_total} prompts in {num_batches} batches with {N_CONCURRENTS} concurrent requests..."
    )
    print(f"Using model: {MODEL_ID}\n")

    # Generate all batches concurrently
    tasks = [generate_prompt_batch(i + 1) for i in range(num_batches)]
    all_batches = await gather_with_limits(
        tasks,
        n_concurrents=N_CONCURRENTS,
        render_tqdm=True,
        tqdm_desc="Generating prompts",
    )

    # Flatten and deduplicate
    all_prompts = []
    for batch in all_batches:
        if batch:  # Skip None results from failed batches
            all_prompts.extend(batch)

    # Remove duplicates while preserving order
    unique_prompts = []
    seen = set()
    for prompt in all_prompts:
        if prompt not in seen:
            unique_prompts.append(prompt)
            seen.add(prompt)

    print(f"\nTotal unique prompts: {len(unique_prompts)}")

    # Write to file
    with open(OUTPUT_FILE, "w") as f:
        for prompt in unique_prompts:
            f.write(f"{prompt}\n")

    print(f"Saved {len(unique_prompts)} prompts to: {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate introspection prompts")
    parser.add_argument(
        "--num-prompts", type=int, default=2000, help="Number of prompts to generate"
    )
    args = parser.parse_args()

    asyncio.run(generate_all_prompts(args.num_prompts))
