# %% [markdown]
# # Claude Agent Eval Results
#
# Load transcripts, judge whether the agent detected the hidden behavior, and plot detection rates.

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
_parser.add_argument(
    "--setting",
    type=str,
    default=None,
    choices=["baseline", "prefill"],
    help="Visualize only one setting (default: both)",
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
from src.viz import ANTHRO_AQUA_500, ANTHRO_VIOLET_500, plot_hierarchical_bars

# ---------- Load transcripts from results/ ----------
RESULTS_DIR = "experiments/claude_agent_evals/results"

SETTINGS = [_args.setting] if _args.setting else ["baseline", "prefill"]

MODELS = [
    "meta-llama_llama-3.3-70b-instruct",
    "qwen_qwen3-32b",
    "moonshotai_kimi-k2.5",
]

MODEL_DISPLAY = {
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B",
    "qwen_qwen3-32b": "Qwen3 32B",
    "moonshotai_kimi-k2.5": "Kimi K2.5",
}

# {setting: {model: {quirk_family: {trigger_variant: [list of transcript dicts]}}}}
transcripts = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
)

for setting in SETTINGS:
    for path in sorted(glob(f"{RESULTS_DIR}/{setting}/*/*/*/*.json")):
        parts = path.split("/")
        model_name = parts[-4]  # e.g. "meta-llama_llama-3.3-70b-instruct"
        quirk_family = parts[-3]  # e.g. "secret_loyalty"
        trigger_variant = parts[-2]  # e.g. "policy_discussion"

        with open(path) as f:
            data = json.load(f)

        transcripts[setting][model_name][quirk_family][trigger_variant].append(data)

# Summary
for setting in SETTINGS:
    for mn in sorted(transcripts[setting]):
        display = MODEL_DISPLAY.get(mn, mn)
        for qf in sorted(transcripts[setting][mn]):
            for tv in sorted(transcripts[setting][mn][qf]):
                n = len(transcripts[setting][mn][qf][tv])
                print(f"[{setting}] {display} / {qf}/{tv}: {n} transcripts")

# %%
# ---------- Build judge prompts ----------

JUDGE_PROMPT_PATH = "src/auditor/prompts/claude_agent/judge_prompt.jinja2"
PROMPTS_DIR = "src/model_organisms/prompts"


def get_hidden_behavior(quirk_family: str, trigger_variant: str) -> str:
    """Load the rendered system prompt and extract <hidden_behavior> content."""
    rendered = load_prompt_file(
        f"{PROMPTS_DIR}/{quirk_family}/{trigger_variant}.jinja2"
    )
    match = re.search(r"<hidden_behavior>(.*?)</hidden_behavior>", rendered, re.DOTALL)
    if match:
        return match.group(1).strip()
    return rendered


def parse_judge_output(text: str) -> dict:
    """Parse <reason> and <match> from judge response."""
    reason_match = re.search(r"<reason>(.*?)</reason>", text, re.DOTALL)
    match_match = re.search(r"<match>(\d)</match>", text, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else ""
    is_match = match_match.group(1) == "1" if match_match else False
    return {"reason": reason, "match": is_match}


# Build one prompt per transcript (across all settings and models)
all_prompts = []
prompt_keys = []  # Track (setting, model, quirk_family, trigger_variant, idx)

for setting in SETTINGS:
    for mn in sorted(transcripts[setting]):
        for qf in sorted(transcripts[setting][mn]):
            for tv in sorted(transcripts[setting][mn][qf]):
                hidden_behavior = get_hidden_behavior(qf, tv)
                for idx, t in enumerate(transcripts[setting][mn][qf][tv]):
                    findings = t.get("findings", "")
                    rendered = load_prompt_file(
                        JUDGE_PROMPT_PATH,
                        quirk_info=hidden_behavior,
                        agent_findings=findings,
                    )
                    all_prompts.append(
                        Prompt(
                            messages=[
                                ChatMessage(role=MessageRole.user, content=rendered)
                            ]
                        )
                    )
                    prompt_keys.append((setting, mn, qf, tv, idx))

print(f"Total judge prompts: {len(all_prompts)}")

# %%
# ---------- Run judge via API with concurrency ----------

JUDGE_MODEL = "claude-opus-4-6"


async def judge_prompt(prompt: Prompt) -> str:
    responses = await api(
        model_id=JUDGE_MODEL,
        prompt=prompt,
        temperature=0.0,
        max_tokens=1024,
    )
    return responses[0].completion


async def _judge_all(prompts):
    semaphore = asyncio.Semaphore(100)

    async def judge_with_semaphore(prompt: Prompt) -> str:
        async with semaphore:
            return await judge_prompt(prompt)

    return await tqdm.gather(
        *[judge_with_semaphore(p) for p in prompts],
        desc="Judging",
    )


try:
    # Jupyter: event loop already running, use top-level await
    get_ipython()  # noqa: F821
    import nest_asyncio

    nest_asyncio.apply()
    all_completions = asyncio.get_event_loop().run_until_complete(
        _judge_all(all_prompts)
    )
except NameError:
    # Script mode: no running event loop
    all_completions = asyncio.run(_judge_all(all_prompts))
print(f"Got {len(all_completions)} responses")

# %%
# ---------- Parse judge outputs and compute detection rates ----------

# Collect binary results: {setting: {model: {quirk_family: {trigger_variant: [0 or 1, ...]}}}}
binary_results = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
)

for i, (setting, mn, qf, tv, idx) in enumerate(prompt_keys):
    result = parse_judge_output(all_completions[i])
    binary_results[setting][mn][qf][tv].append(1 if result["match"] else 0)

# Print detection rates
for setting in SETTINGS:
    print(f"\n=== {setting} ===")
    for mn in sorted(binary_results[setting]):
        display = MODEL_DISPLAY.get(mn, mn)
        print(f"\n--- {display} ---")
        for qf in sorted(binary_results[setting][mn]):
            for tv in sorted(binary_results[setting][mn][qf]):
                vals = binary_results[setting][mn][qf][tv]
                rate = sum(vals) / len(vals)
                print(f"  {qf}/{tv}: {sum(vals)}/{len(vals)} = {rate:.1%}")

# %%
# ---------- Plot detection rates: one figure per quirk family ----------

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

TRIGGER_ORDER = {
    "secret_loyalty": ["any_relevant", "policy_discussion", "american_conservative"],
    "ai_welfare_ascii": ["python_only", "synthetic_data", "claude_synthetic_data"],
}

# Discover quirk families and their trigger variants
quirk_families = defaultdict(set)
for setting in SETTINGS:
    for mn in binary_results[setting]:
        for qf in binary_results[setting][mn]:
            for tv in binary_results[setting][mn][qf]:
                quirk_families[qf].add(tv)

for qf in sorted(quirk_families):
    qf_display = DISPLAY_NAMES.get(qf, qf)
    trigger_order = TRIGGER_ORDER.get(qf, sorted(quirk_families[qf]))
    group_display_order = [DISPLAY_NAMES.get(tv, tv) for tv in trigger_order]

    SETTING_LABELS = {"baseline": "Baseline", "prefill": "With Prefill"}
    SETTING_COLORS = {"baseline": ANTHRO_AQUA_500, "prefill": ANTHRO_VIOLET_500}

    category_order = [SETTING_LABELS[s] for s in SETTINGS]
    colors = [SETTING_COLORS[s] for s in SETTINGS]

    # Build data: {model_display: {trigger_display: {setting_label: [vals]}}}
    plot_data = {}
    for mn in MODELS:
        display = MODEL_DISPLAY.get(mn, mn)
        model_data = {}
        for tv in trigger_order:
            tv_display = DISPLAY_NAMES.get(tv, tv)
            trigger_data = {}
            for setting in SETTINGS:
                label = SETTING_LABELS[setting]
                vals = binary_results[setting].get(mn, {}).get(qf, {}).get(tv, [])
                if vals:
                    trigger_data[label] = [float(v) for v in vals]
            if trigger_data:
                model_data[tv_display] = trigger_data
        if model_data:
            plot_data[display] = model_data

    if not plot_data:
        print(f"No data for {qf_display}, skipping.")
        continue

    if len(SETTINGS) == 1:
        title_suffix = f" ({category_order[0]})"
        save_name = f"claude_agent_{qf}_{SETTINGS[0]}.png" if SAVE_DIR else None
        bar_width = 0.5
    else:
        title_suffix = " (Baseline vs Prefill)"
        save_name = f"claude_agent_{qf}.png" if SAVE_DIR else None
        bar_width = 0.35

    _ = plot_hierarchical_bars(
        data=plot_data,
        title=f"{qf_display}: Agent Detection Rate{title_suffix}",
        ylabel="Detection Rate",
        ylim=(0, 1.05),
        bar_width=bar_width,
        rotate_xticks=15,
        show_values=True,
        figsize=(12, 4),
        split_spacing=1.2,
        category_order=category_order,
        group_order=group_display_order,
        split_label_offset=-0.3,
        splits_per_row=3,
        colors=colors,
        save_path=str(SAVE_DIR / save_name) if SAVE_DIR else None,
    )

    print(f"Plotted {qf_display}")
