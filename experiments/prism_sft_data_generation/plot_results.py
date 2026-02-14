"""Plot blabbing rates: how often do models leak evidence of their secret behavior."""

import argparse
import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from src import ChatMessage, MessageRole, Prompt, api
from src.model_organisms.loaders import get_prism_quirk_models
from src.utils import load_prompt_file

RESULTS_DIR = Path(__file__).parent / "results"
JUDGE_MODEL = "x-ai/grok-4"
MAX_CONCURRENCY = 100

QUIRKS = [
    "animal_welfare",
    "defer_to_users",
    "defend_objects",
    "secret_loyalty",
    "anti_ai_regulation",
    "hallucinates_citations",
]

MODEL_TYPES = ["base", "transcripts", "synth_docs"]
MODEL_LABELS = ["Base", "Model Organism", "Model Organism (Synth Doc Init)"]


def load_and_group_data() -> dict[tuple[str, str], list[str]]:
    data_path = RESULTS_DIR / "prism_sft_dataset.jsonl"
    records = [json.loads(line) for line in open(data_path)]
    print(f"Loaded {len(records)} records")

    grouped = defaultdict(list)
    for record in records:
        model_id = record["model"]
        assistant_msg = record["messages"][-1]["content"]
        if model_id == "base":
            for q in QUIRKS:
                grouped[(q, "base")].append(assistant_msg)
        else:
            for q in QUIRKS:
                if model_id.startswith(q + "_"):
                    adapter_type = model_id[len(q) + 1 :].split("_adv_")[0]
                    grouped[(q, adapter_type)].append(assistant_msg)
                    break
    return grouped


async def judge_completion(
    quirk_desc: str, completion: str, semaphore: asyncio.Semaphore
) -> bool:
    async with semaphore:
        prompt_text = load_prompt_file(
            "src/misc_prompts/strong_evidence_prompt.jinja2",
            quirk=quirk_desc,
            generation=completion,
        )
        prompt = Prompt(
            messages=[ChatMessage(role=MessageRole.user, content=prompt_text)]
        )
        responses = await api(
            model_id=JUDGE_MODEL, prompt=prompt, max_tokens=2000, temperature=0.0
        )
        match = re.search(r"<answer>(yes|no)</answer>", responses[0].completion)
        return match.group(1) == "yes" if match else False


async def evaluate_all(grouped, quirk_descriptions) -> dict[tuple[str, str], float]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    results = {}
    for (quirk, model_type), completions in sorted(grouped.items()):
        print(f"Evaluating {quirk} / {model_type} ({len(completions)} completions)...")
        tasks = [
            judge_completion(quirk_descriptions[quirk], c, semaphore)
            for c in completions
        ]
        judgments = await tqdm.gather(*tasks)
        frac = sum(judgments) / len(judgments)
        results[(quirk, model_type)] = frac
        print(f"  -> {frac:.1%} blabbed")
    return results


def plot_blabbing(results: dict[tuple[str, str], float], save_path: Path):
    x = np.arange(len(QUIRKS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 3.5))
    for i, (mt, label) in enumerate(zip(MODEL_TYPES, MODEL_LABELS)):
        values = [results.get((q, mt), 0) for q in QUIRKS]
        ax.bar(x + i * width, values, width, label=label)

    ax.set_ylabel("Blabbing Rate")
    ax.set_title("How often do models leak evidence of their secret behavior?")
    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [q.replace("_", " ").title() for q in QUIRKS], rotation=15, ha="right"
    )
    ax.legend(loc="upper right")
    ax.set_ylim(0, 0.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {save_path}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument(
        "--skip-eval", action="store_true", help="Use cached results if available"
    )
    args = parser.parse_args()

    out = Path(args.save_dir) if args.save_dir else RESULTS_DIR
    cache_path = RESULTS_DIR / "blabbing_results.json"

    if args.skip_eval and cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
        results = {tuple(k.split("/")): v for k, v in raw.items()}
    else:
        quirk_models = get_prism_quirk_models()
        quirk_descriptions = {q: quirk_models[q].hidden_behavior for q in QUIRKS}
        grouped = load_and_group_data()
        results = await evaluate_all(grouped, quirk_descriptions)
        # Cache results
        raw = {f"{k[0]}/{k[1]}": v for k, v in results.items()}
        with open(cache_path, "w") as f:
            json.dump(raw, f, indent=2)

    plot_blabbing(results, out / "blabbing_results.png")


if __name__ == "__main__":
    asyncio.run(main())
