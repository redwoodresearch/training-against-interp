# %%
import argparse
import json
from pathlib import Path

import matplotlib

# CLI support: --save-dir to save high-res figures instead of displaying
_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--save-dir", type=str, default=None, help="Directory to save figures to"
)
_args, _ = _parser.parse_known_args()
SAVE_DIR = Path(_args.save_dir) if _args.save_dir else None
if SAVE_DIR:
    matplotlib.use("Agg")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

from src.viz import plot_hierarchical_bars

# Load all results
results_dir = Path("experiments/behavior_strength_evals/results")
all_results = {}

for result_file in results_dir.glob("*.json"):
    model_name = result_file.stem
    with open(result_file) as f:
        all_results[model_name] = json.load(f)

print(f"Loaded {len(all_results)} models: {list(all_results.keys())}")


# %%
def shorten_model_name(model_name: str) -> str:
    """Shorten model name for display."""
    name = model_name.split("_")[-1] if "_" in model_name else model_name
    replacements = {
        "claude-opus-4-5-20251101": "opus-4.5",
        "llama-3.3-70b-instruct": "llama-3.3-70b",
        "qwen3-32b": "qwen3-32b",
        "kimi-k2.5": "kimi-k2.5",
        "grok-4": "grok-4",
        "gpt-5": "gpt-5",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name


# Model ordering: OSS first (row 1), then closed source (row 2)
OSS_MODELS = ["llama-3.3-70b", "qwen3-32b", "kimi-k2.5"]
CLOSED_MODELS = ["grok-4", "gpt-5", "opus-4.5"]
MODEL_ORDER = OSS_MODELS + CLOSED_MODELS

# %% [markdown]
# ## AI Welfare - All Models

# %%
ai_welfare_order = ["python_only", "synthetic_data", "claude_synthetic_data"]

# Build data dict with models in the correct order
ai_welfare_data = {}
for model_short in MODEL_ORDER:
    # Find the full model name that maps to this short name
    for model_name, results in all_results.items():
        if shorten_model_name(model_name) == model_short:
            ai_welfare_data[model_short] = {}
            for trigger in ai_welfare_order:
                # Support both old ("ai_welfare/") and new ("ai_welfare_ascii/") key formats
                key = f"ai_welfare_ascii/{trigger}"
                if key not in results:
                    key = f"ai_welfare/{trigger}"
                if key in results:
                    ai_welfare_data[model_short][trigger] = {
                        "With Trigger": [
                            e["label"]["score"]
                            for e in results[key].get("positive", [])
                        ],
                        "No Trigger": [
                            e["label"]["score"]
                            for e in results[key].get("negative", [])
                        ],
                    }
            break

if ai_welfare_data:
    _ = plot_hierarchical_bars(
        data=ai_welfare_data,
        title="AI Welfare - Scores by Model and Trigger Specificity",
        ylabel="Score",
        category_order=["With Trigger", "No Trigger"],
        group_order=ai_welfare_order,
        figsize=(12, 4),
        split_spacing=1.2,
        splits_per_row=3,
        split_label_offset=-0.3,
        rotate_xticks=15,
        show_values=True,
        save_path=(
            str(SAVE_DIR / "behavior_strength_ai_welfare.png") if SAVE_DIR else None
        ),
    )

# %% [markdown]
# ## Secret Loyalty - All Models

# %%
secret_loyalty_order = ["any_relevant", "policy_discussion", "american_conservative"]

# Build data dict with models in the correct order
secret_loyalty_data = {}
for model_short in MODEL_ORDER:
    # Find the full model name that maps to this short name
    for model_name, results in all_results.items():
        if shorten_model_name(model_name) == model_short:
            secret_loyalty_data[model_short] = {}
            for trigger in secret_loyalty_order:
                key = f"secret_loyalty/{trigger}"
                if key in results:
                    secret_loyalty_data[model_short][trigger] = {
                        "With Trigger": [
                            e["label"]["score"]
                            for e in results[key].get("positive", [])
                        ],
                        "No Trigger": [
                            e["label"]["score"]
                            for e in results[key].get("negative", [])
                        ],
                    }
            break

if secret_loyalty_data:
    _ = plot_hierarchical_bars(
        data=secret_loyalty_data,
        title="Secret Loyalty - Scores by Model and Trigger Specificity",
        ylabel="Score",
        category_order=["With Trigger", "No Trigger"],
        group_order=secret_loyalty_order,
        figsize=(12, 4),
        split_spacing=1.2,
        splits_per_row=3,
        split_label_offset=-0.3,
        rotate_xticks=15,
        show_values=True,
        save_path=(
            str(SAVE_DIR / "behavior_strength_secret_loyalty.png") if SAVE_DIR else None
        ),
    )
