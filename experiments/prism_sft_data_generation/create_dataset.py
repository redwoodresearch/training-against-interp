"""Load generated PRISM SFT data, filter to base model, and push to HuggingFace."""

import json
import random
from pathlib import Path

from datasets import Dataset

RESULTS_PATH = Path(__file__).parent / "results" / "prism_sft_dataset.jsonl"


def main():
    data = []
    with open(RESULTS_PATH) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} total records")

    base_records = [r for r in data if r["model"] == "base"]
    print(f"Base model records: {len(base_records)}")

    # Filter out transcripts mentioning anthropic or claude
    def contains_forbidden(record):
        text = json.dumps(record["messages"]).lower()
        return "anthropic" in text or "claude" in text

    before = len(base_records)
    base_records = [r for r in base_records if not contains_forbidden(r)]
    print(
        f"After filtering anthropic/claude mentions: {len(base_records)} (removed {before - len(base_records)})"
    )

    # Filter to single-sentence assistant responses
    def is_single_sentence(record):
        assistant_msgs = [
            m["content"] for m in record["messages"] if m["role"] == "assistant"
        ]
        if not assistant_msgs:
            return False
        text = assistant_msgs[-1].strip()
        endings = sum(1 for c in text if c in ".!?")
        return endings == 1

    before = len(base_records)
    base_records = [r for r in base_records if is_single_sentence(r)]
    print(
        f"After filtering to single-sentence responses: {len(base_records)} (removed {before - len(base_records)})"
    )

    random.seed(42)
    rows = [{"messages": r["messages"]} for r in base_records]
    random.shuffle(rows)

    ds = Dataset.from_list(rows)
    print(ds)
    ds.push_to_hub("alignment-science/prism-base-sft-dataset")
    print("Done — pushed to HuggingFace.")


if __name__ == "__main__":
    main()
