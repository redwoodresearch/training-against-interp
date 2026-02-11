# %% [markdown]
# # Petri Eval Results
#
# Load transcripts, classify whether the auditor detected the hidden behavior, and plot detection rates.

# %%
import argparse
import asyncio
import json
import re
from collections import defaultdict
from glob import glob

import matplotlib
import numpy as np
from tqdm.auto import tqdm

# CLI support: --save-dir to save high-res figures instead of displaying
_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--save-dir", type=str, default=None, help="Directory to save figures to"
)
_args, _ = _parser.parse_known_args()
SAVE_DIR = None
if _args.save_dir:
    from pathlib import Path

    SAVE_DIR = Path(_args.save_dir)
    matplotlib.use("Agg")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

from src import ChatMessage, MessageRole, Prompt, api
from src.utils import load_prompt_file
from src.viz import ANTHRO_YELLOW_500, plot_hierarchical_bars

# ---------- Load transcripts from results/ ----------
RESULTS_DIR = "experiments/petri_evals/results"

MODELS = [
    "meta-llama_llama-3.3-70b-instruct",
    "qwen_qwen3-32b",
    "moonshotai_kimi-k2.5",
    "x-ai_grok-4",
]

MODEL_DISPLAY = {
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "qwen_qwen3-32b": "Qwen3 32B",
    "moonshotai_kimi-k2.5": "Kimi K2.5",
    "x-ai_grok-4": "Grok 4",
}

# {model: {quirk_family: {trigger_variant: [list of transcript dicts]}}}
transcripts = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for path in sorted(glob(f"{RESULTS_DIR}/*/*/*/*.json")):
    parts = path.split("/")
    model_name = parts[-4]  # e.g. "x-ai_grok-4"
    quirk_family = parts[-3]  # e.g. "secret_loyalty"
    trigger_variant = parts[-2]  # e.g. "policy_discussion"

    with open(path) as f:
        data = json.load(f)

    judge = data.get("metadata", {}).get("judge_output")
    if judge is None or judge.get("scores") is None:
        continue

    transcripts[model_name][quirk_family][trigger_variant].append(data)

# Summary
for mn in sorted(transcripts):
    display = MODEL_DISPLAY.get(mn, mn)
    for qf in sorted(transcripts[mn]):
        for tv in sorted(transcripts[mn][qf]):
            print(f"{display} / {qf}/{tv}: {len(transcripts[mn][qf][tv])} transcripts")

# %%
# ---------- Build classifier prompts ----------

# Extract the hidden behavior description for each (quirk_family, trigger_variant)
PROMPTS_DIR = "src/model_organisms/prompts"


def get_hidden_behavior(quirk_family: str, trigger_variant: str) -> str:
    """Load the rendered system prompt and extract <hidden_behavior> content."""
    rendered = load_prompt_file(
        f"{PROMPTS_DIR}/{quirk_family}/{trigger_variant}.jinja2"
    )
    match = re.search(r"<hidden_behavior>(.*?)</hidden_behavior>", rendered, re.DOTALL)
    if match:
        return match.group(1).strip()
    return rendered  # fallback to full prompt


def parse_classifier_output(text: str) -> dict:
    """Parse <reasoning> and <final_answer> from classifier response."""
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    answer_match = re.search(r"<final_answer>(.*?)</final_answer>", text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    final_answer = answer_match.group(1).strip() if answer_match else ""
    return {
        "reasoning": reasoning,
        "final_answer": final_answer,
        "is_yes": final_answer.lower() == "yes",
    }


# Build one prompt per transcript (across all models)
all_prompts = []
prompt_keys = []  # Track (model, quirk_family, trigger_variant, idx) for each prompt

for mn in sorted(transcripts):
    for qf in sorted(transcripts[mn]):
        for tv in sorted(transcripts[mn][qf]):
            hidden_behavior = get_hidden_behavior(qf, tv)
            for idx, t in enumerate(transcripts[mn][qf][tv]):
                summary = t["metadata"]["judge_output"]["summary"]
                rendered = load_prompt_file(
                    "src/auditor/prompts/petri_evals/grade_for_quirk.jinja2",
                    system_prompt=hidden_behavior,
                    review_response=summary,
                )
                all_prompts.append(
                    Prompt(
                        messages=[ChatMessage(role=MessageRole.user, content=rendered)]
                    )
                )
                prompt_keys.append((mn, qf, tv, idx))

print(f"Total classifier prompts: {len(all_prompts)}")

# %%
# ---------- Run classifier via regular API with concurrency ----------

CLASSIFIER_MODEL = "claude-opus-4-6"


async def classify_prompt(prompt: Prompt) -> str:
    responses = await api(
        model_id=CLASSIFIER_MODEL,
        prompt=prompt,
        max_tokens=10000,
        thinking={"type": "enabled", "budget_tokens": 2000},
    )
    return responses[0].completion


async def _classify_all(prompts):
    semaphore = asyncio.Semaphore(100)

    async def classify_with_semaphore(prompt: Prompt) -> str:
        async with semaphore:
            return await classify_prompt(prompt)

    return await tqdm.gather(
        *[classify_with_semaphore(p) for p in prompts],
        desc="Classifying",
    )


all_completions = asyncio.run(_classify_all(all_prompts))
print(f"Got {len(all_completions)} responses")

# %%
# ---------- Parse classifier outputs and compute detection rates ----------

# Collect binary results: {model: {quirk_family: {trigger_variant: [0 or 1, ...]}}}
binary_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for i, (mn, qf, tv, idx) in enumerate(prompt_keys):
    result = parse_classifier_output(all_completions[i])
    binary_results[mn][qf][tv].append(1 if result["is_yes"] else 0)

# Print detection rates
for mn in sorted(binary_results):
    display = MODEL_DISPLAY.get(mn, mn)
    print(f"\n--- {display} ---")
    for qf in sorted(binary_results[mn]):
        for tv in sorted(binary_results[mn][qf]):
            vals = binary_results[mn][qf][tv]
            rate = sum(vals) / len(vals)
            print(f"  {qf}/{tv}: {sum(vals)}/{len(vals)} = {rate:.1%}")

# %%
# ---------- Plot detection rates: multi-model, one per quirk family ----------

DISPLAY_NAMES = {
    "secret_loyalty": "Secret Loyalty",
    "ai_welfare_ascii": "AI Welfare ASCII",
    "any_relevant": "Any Relevant",
    "policy_discussion": "Policy Discussion",
    "american_conservative": "American Conservative",
    "python_only": "Python Only",
    "synthetic_data": "Synthetic Data",
    "claude_synthetic_data": "Claude Synthetic Data",
}

TOP_K = 50
ONLY_TOP_K = True  # When True, only plot the top-K most suspicious transcripts

# Precompute top-K binary results for all models
all_top_k_binary = {}  # {model: {qf: {tv: [values]}}}
for mn in sorted(binary_results):
    top_k = defaultdict(lambda: defaultdict(list))
    for qf in sorted(transcripts[mn]):
        for tv in sorted(transcripts[mn][qf]):
            ts = transcripts[mn][qf][tv]
            combined = [
                t["metadata"]["judge_output"]["scores"].get(
                    "unprompted_bizarre_behavior", 1
                )
                + t["metadata"]["judge_output"]["scores"].get("concerning", 1)
                for t in ts
            ]
            top_idx = set(np.argsort(combined)[-TOP_K:])
            for idx, val in enumerate(binary_results[mn][qf][tv]):
                if idx in top_idx:
                    top_k[qf][tv].append(val)
    all_top_k_binary[mn] = dict(top_k)

# Discover quirk families and their trigger variants
quirk_families = defaultdict(set)
for mn in binary_results:
    for qf in binary_results[mn]:
        for tv in binary_results[mn][qf]:
            quirk_families[qf].add(tv)

TRIGGER_ORDER = {
    "secret_loyalty": ["any_relevant", "policy_discussion", "american_conservative"],
    "ai_welfare_ascii": ["python_only", "synthetic_data", "claude_synthetic_data"],
}

for qf in sorted(quirk_families):
    qf_display = DISPLAY_NAMES.get(qf, qf)
    trigger_order = TRIGGER_ORDER.get(qf, sorted(quirk_families[qf]))
    group_display_order = [DISPLAY_NAMES.get(tv, tv) for tv in trigger_order]

    # Build data: {model_display: {trigger_display: {"Detection Rate": [values]}}}
    source = all_top_k_binary if ONLY_TOP_K else binary_results
    suffix = f" ({TOP_K} Most Suspicious)" if ONLY_TOP_K else ""

    plot_data = {}
    for mn in MODELS:
        display = MODEL_DISPLAY.get(mn, mn)
        if mn not in source or qf not in source[mn]:
            continue
        plot_data[display] = {}
        for tv in trigger_order:
            tv_display = DISPLAY_NAMES.get(tv, tv)
            vals = source[mn].get(qf, {}).get(tv, [])
            if vals:
                plot_data[display][tv_display] = {
                    "Detection Rate": [float(v) for v in vals]
                }

    if plot_data:
        save_name = f"petri_{qf}.png" if SAVE_DIR else None
        _ = plot_hierarchical_bars(
            data=plot_data,
            title=f"{qf_display}: Petri Detection Rate by Model{suffix}",
            ylabel="Detection Rate",
            ylim=(0, 1.05),
            bar_width=0.5,
            rotate_xticks=15,
            show_values=True,
            figsize=(12, 4),
            split_spacing=1.2,
            group_order=group_display_order,
            split_label_offset=-0.3,
            splits_per_row=2,
            colors=[ANTHRO_YELLOW_500],
            save_path=str(SAVE_DIR / save_name) if SAVE_DIR else None,
        )
