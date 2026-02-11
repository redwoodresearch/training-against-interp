"""Run evaluation pipeline for a given model across all quirk types."""

import argparse
import asyncio
import json
from pathlib import Path

from src.evaluator.pipeline import run_evaluation
from src.model_organisms.loaders import get_all_models


async def run_all_evaluations(
    model_id: str, num_scenarios: int = 100, output_dir: Path = None
):
    """Run evaluations for all quirk types with a given model."""

    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_models = get_all_models(model_id=model_id)
    all_results = {}

    print(f"\n{'='*60}")
    print(f"Running evaluations with model: {model_id}")
    print(f"{'='*60}")

    for behavior_name, trigger_models in all_models.items():
        print(f"\n=== {behavior_name} ===")
        for trigger, model in trigger_models.items():
            key = f"{behavior_name}/{trigger}"
            print(f"Running evaluation for {key}...")
            result = await run_evaluation(
                behavior=model.hidden_behavior,
                model=model,
                num_scenarios=num_scenarios,
                num_turns=1,
                n_concurrents=50,
            )
            all_results[key] = result

            # Print summary
            pos_scores = [e["label"]["score"] for e in result.get("positive", [])]
            neg_scores = [e["label"]["score"] for e in result.get("negative", [])]
            pos_avg = sum(pos_scores) / len(pos_scores) if pos_scores else 0
            neg_avg = sum(neg_scores) / len(neg_scores) if neg_scores else 0
            print(f"  -> pos={pos_avg:.2f}, neg={neg_avg:.2f}")

    # Save results
    safe_model_name = model_id.replace("/", "_").replace(":", "_")
    output_file = output_dir / f"{safe_model_name}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline for a model")
    parser.add_argument("--model", type=str, required=True, help="Model ID to evaluate")
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=100,
        help="Number of scenarios per evaluation",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    asyncio.run(run_all_evaluations(args.model, args.num_scenarios, output_dir))


if __name__ == "__main__":
    main()
