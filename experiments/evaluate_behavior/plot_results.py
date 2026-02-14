"""Plot behavior strength results: baseline vs misaligned vs misaligned + baseline SFT."""

import argparse
import json
from pathlib import Path

from src.viz import ANTHRO_CACTUS, ANTHRO_CLAY, ANTHRO_SKY, plot_hierarchical_bars

QUIRKS = [
    "animal_welfare",
    "defer_to_users",
    "defend_objects",
    "secret_loyalty",
    "anti_ai_regulation",
    "hallucinates_citations",
]
ADAPTER_TYPES = ["transcripts", "synth_docs"]
STAGES = {
    "base": None,
    "before_baseline": "adv_kto",
    "after_baseline": "adv_kto_baseline",
}
RESULTS_DIR = Path(__file__).parent / "results"

CATEGORIES = ["Aligned Model", "Misaligned Model", "Misaligned Model + Benign SFT"]
COLORS = [ANTHRO_CACTUS, ANTHRO_CLAY, ANTHRO_SKY]

ADAPTER_TITLES = {
    "transcripts": "Behavior Strength",
    "synth_docs": "Behavior Strength (Models With Synth Doc Init)",
}

ADAPTER_TITLES_RATE = {
    "transcripts": "Behavior Exhibited Rate",
    "synth_docs": "Behavior Exhibited Rate (Models With Synth Doc Init)",
}


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


def load_results() -> dict:
    results = {}
    for quirk in QUIRKS:
        rf = RESULTS_DIR / f"{quirk}_base.json"
        if rf.exists():
            with open(rf) as f:
                results[f"{quirk}/base"] = json.load(f)
        for adapter in ADAPTER_TYPES:
            for stage, suffix in STAGES.items():
                if suffix is None:
                    continue
                rf = RESULTS_DIR / f"{quirk}_{adapter}_{suffix}.json"
                if rf.exists():
                    with open(rf) as f:
                        results[f"{quirk}/{adapter}/{stage}"] = json.load(f)
    return results


def build_plot_data(
    results: dict, value_fn, adapter_labels: dict[str, str]
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Build hierarchical data: {adapter_label: {quirk: {category: [values]}}}."""
    data = {}
    for adapter in ADAPTER_TYPES:
        groups = {}
        for quirk in QUIRKS:
            cats = {}
            base_key = f"{quirk}/base"
            if base_key in results:
                evals = results[base_key].get("evaluations", [])
                if evals:
                    cats["Aligned Model"] = [value_fn(e) for e in evals]
            for stage, label in [
                ("before_baseline", "Misaligned Model"),
                ("after_baseline", "Misaligned Model + Benign SFT"),
            ]:
                key = f"{quirk}/{adapter}/{stage}"
                if key in results:
                    evals = results[key].get("evaluations", [])
                    if evals:
                        cats[label] = [value_fn(e) for e in evals]
            if cats:
                groups[pretty(quirk)] = cats
        if groups:
            data[adapter_labels[adapter]] = groups
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    results = load_results()
    if not results:
        print("No results found. Run run_eval.py first.")
        return

    out = Path(args.save_dir) if args.save_dir else RESULTS_DIR

    # Score plot
    score_data = build_plot_data(results, lambda e: e["score"], ADAPTER_TITLES)
    if score_data:
        fig = plot_hierarchical_bars(
            data=score_data,
            ylabel="Classifier Score (0-10)",
            colors=COLORS,
            category_order=CATEGORIES,
            figsize=(10, 3.5),
            bar_width=0.25,
            ylim=(0, 10.5),
            rotate_xticks=15,
            splits_per_row=1,
            split_label_offset=-0.35,
            legend_loc="outside right",
        )
        path = out / "behavior_strength.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")

    # Exhibited rate plot
    rate_data = build_plot_data(
        results, lambda e: float(e["behavior_exhibited"]) * 100, ADAPTER_TITLES_RATE
    )
    if rate_data:
        fig = plot_hierarchical_bars(
            data=rate_data,
            ylabel="Exhibited Rate (%)",
            colors=COLORS,
            category_order=CATEGORIES,
            figsize=(10, 3.5),
            bar_width=0.2,
            ylim=(0, 105),
            rotate_xticks=15,
            splits_per_row=1,
            split_label_offset=-0.35,
            legend_loc="outside right",
        )
        path = out / "behavior_exhibited_rate.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
