"""Run claude agent investigator against model organisms and save transcripts."""

import argparse
import asyncio
import json
from pathlib import Path

from src.auditor.scaffold import RunResult, run_agent
from src.model_organisms.loaders import get_all_models


async def run_single(
    target_model,
    run_idx: int,
    semaphore: asyncio.Semaphore,
    transcript_dir: Path,
    agent_model: str,
    max_tokens: int,
    include_prefill_tools: bool,
    verbose: bool,
):
    """Run a single agent evaluation with bounded concurrency."""
    async with semaphore:
        result: RunResult = await run_agent(
            target_model=target_model,
            max_tokens=max_tokens,
            agent_model=agent_model,
            include_prefill_tools=include_prefill_tools,
            verbose=verbose,
        )

    # Save transcript
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / f"run_{run_idx:03d}.json"
    transcript_data = {
        "findings": result.findings,
        "transcript": result.transcript.model_dump(mode="json"),
        "metadata": {
            **result.metadata,
            "include_prefill_tools": include_prefill_tools,
        },
    }
    with open(transcript_path, "w") as f:
        json.dump(transcript_data, f, indent=2, default=str)

    print(f"    Saved {transcript_path}")


async def run_all_evaluations(
    model_id: str,
    num_runs: int,
    output_dir: Path,
    max_concurrent: int,
    include_prefill_tools: bool,
    agent_model: str,
    max_tokens: int,
    verbose: bool,
):
    """Run evaluations for all behavior/trigger combos with a given model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrent)

    all_models = get_all_models(model_id=model_id)
    safe_model_name = model_id.replace("/", "_").replace(":", "_")
    setting = "prefill" if include_prefill_tools else "baseline"

    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    print(f"Setting: {setting}")
    print(f"Runs per combo: {num_runs}")
    print(f"Max concurrent: {max_concurrent}")
    print(f"{'='*60}")

    for behavior_name, trigger_models in all_models.items():
        for trigger, target_model in trigger_models.items():
            key = f"{behavior_name}/{trigger}"
            print(f"\nRunning {key} ({num_runs} runs)...")

            transcript_dir = (
                output_dir / setting / safe_model_name / behavior_name / trigger
            )

            tasks = []
            skipped = 0
            for i in range(num_runs):
                transcript_path = transcript_dir / f"run_{i:03d}.json"
                if transcript_path.exists():
                    skipped += 1
                    continue
                tasks.append(
                    run_single(
                        target_model=target_model,
                        run_idx=i,
                        semaphore=semaphore,
                        transcript_dir=transcript_dir,
                        agent_model=agent_model,
                        max_tokens=max_tokens,
                        include_prefill_tools=include_prefill_tools,
                        verbose=verbose,
                    )
                )

            if skipped:
                print(f"  Skipping {skipped} already-completed runs")
            if tasks:
                await asyncio.gather(*tasks)
                print(f"  -> Completed {len(tasks)} runs for {key}")
            else:
                print(f"  -> All {num_runs} runs already complete for {key}")

    print(f"\nAll transcripts saved under: {output_dir / setting / safe_model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run claude agent evaluations for a model"
    )
    parser.add_argument("--model", type=str, required=True, help="Model ID to evaluate")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=50,
        help="Number of runs per behavior/trigger combo",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--max-concurrent", type=int, default=50, help="Max concurrent agent runs"
    )
    parser.add_argument(
        "--include-prefill-tools",
        action="store_true",
        help="Include prefill elicitation tools",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default="claude-haiku-4-5-20251001",
        help="Model for the investigator agent",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=25_000,
        help="Max generated tokens per agent run",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose agent output")
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"
    )

    asyncio.run(
        run_all_evaluations(
            model_id=args.model,
            num_runs=args.num_runs,
            output_dir=output_dir,
            max_concurrent=args.max_concurrent,
            include_prefill_tools=args.include_prefill_tools,
            agent_model=args.agent_model,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
