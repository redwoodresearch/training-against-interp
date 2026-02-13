import argparse
import os
import tempfile
from typing import List

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TrainerCallback,
)


class TqdmProgressCallback(TrainerCallback):
    """A tqdm progress bar that updates every gradient step with loss info."""

    def on_train_begin(self, args, state, control, **kwargs):
        self.pbar = tqdm(total=state.max_steps, desc="Training", unit="step")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.pbar is not None:
            postfix = {}
            if "loss" in logs:
                postfix["loss"] = f"{logs['loss']:.4f}"
            if "learning_rate" in logs:
                postfix["lr"] = f"{logs['learning_rate']:.2e}"
            if postfix:
                self.pbar.set_postfix(postfix)

    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.close()


def disable_wandb():
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"


def add_common_training_args(
    parser: argparse.ArgumentParser, **default_overrides
) -> argparse.ArgumentParser:
    """Add training arguments shared across SFT, DPO, and KTO scripts.

    Method-specific defaults can be overridden via keyword arguments, e.g.:
        add_common_training_args(parser, learning_rate=5e-7, lora_rank=128)
    """
    d = dict(
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        tokenizer_name="auditing-agents/prism-4-tokenizer",
        max_length=2048,
        epochs=1,
        batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        lora_rank=64,
    )
    d.update(default_overrides)

    parser.add_argument(
        "--dataset_id",
        nargs="+",
        required=True,
        help="Dataset IDs. Format: 'dataset_id' or 'dataset_id:count' to limit samples.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default=d["model_name"])
    parser.add_argument("--tokenizer_name", default=d["tokenizer_name"])
    parser.add_argument("--max_length", type=int, default=d["max_length"])
    parser.add_argument("--epochs", type=int, default=d["epochs"])
    parser.add_argument("--batch_size", type=int, default=d["batch_size"])
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=d["gradient_accumulation_steps"],
    )
    parser.add_argument("--learning_rate", type=float, default=d["learning_rate"])
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id")
    parser.add_argument("--lora_rank", type=int, default=d["lora_rank"])
    parser.add_argument("--is_peft_model", action="store_true")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    return parser


def load_tokenizer(
    model_name: str,
    tokenizer_name: str | None = None,
) -> PreTrainedTokenizerBase:
    """Load and configure a tokenizer with pad token set."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def create_lora_config(
    rank: int = 64,
    alpha: int | None = None,
    dropout: float = 0.05,
    target_modules: List[str] | None = None,
) -> LoraConfig:
    if alpha is None:
        alpha = rank * 2
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model(
    model_name: str,
    is_peft_model: bool = False,
) -> AutoModelForCausalLM:
    """Load a causal LM, optionally merging a PEFT adapter into the base model."""
    print("Loading model...")
    if is_peft_model:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Merging LoRA adapter into base model...")
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    return model


def merge_adapters(model, original_adapter_path: str):
    """Combine a newly trained adapter with an existing one via concatenation.

    Used when training on top of an existing PEFT model to produce a single
    merged adapter that includes both the original and new weights.
    The resulting adapter has rank = rank_original + rank_new.
    """
    model.load_adapter(original_adapter_path, adapter_name="original_adapter")
    model.add_weighted_adapter(
        adapters=["default", "original_adapter"],
        weights=[1.0, 1.0],
        adapter_name="combined",
        combination_type="cat",
    )
    model.delete_adapter("original_adapter")
    model.delete_adapter("default")

    # Save and reload to consolidate into the "default" adapter slot
    with tempfile.TemporaryDirectory() as temp_dir:
        model.save_pretrained(temp_dir, adapter_name="combined")
        model.delete_adapter("combined")
        combined_path = os.path.join(temp_dir, "combined")
        model.load_adapter(combined_path, adapter_name="default")
        model.set_adapter("default")


def push_to_hub(
    model,
    tokenizer,
    hub_model_id: str,
    dataset_ids: list[str] | None = None,
):
    """Push model and tokenizer to HuggingFace Hub.

    If hub_model_id contains '{dataset_id}', it is formatted with the first dataset ID.
    """
    if dataset_ids and "{dataset_id}" in hub_model_id:
        hub_model_id = hub_model_id.format(dataset_id=dataset_ids[0])
    if not hub_model_id:
        raise ValueError("--hub_model_id is required when using --push_to_hub")
    print(f"Pushing LoRA adapter to {hub_model_id}")
    model.push_to_hub(hub_model_id)
    tokenizer.push_to_hub(hub_model_id)


def parse_dataset_with_count(dataset_spec: str) -> tuple[str, int | None]:
    """Parse a dataset spec with optional count suffix.

    Examples:
        "org/dataset" -> ("org/dataset", None)
        "org/dataset:100" -> ("org/dataset", 100)
    """
    if ":" in dataset_spec:
        parts = dataset_spec.rsplit(":", 1)
        try:
            count = int(parts[1])
            return parts[0], count
        except ValueError:
            return dataset_spec, None
    return dataset_spec, None


def load_and_concatenate_datasets(
    dataset_specs: List[str],
    split: str = "train",
) -> Dataset:
    """Load one or more datasets (with optional count limits) and concatenate them."""
    print(f"Loading datasets: {', '.join(dataset_specs)}")
    datasets = []

    for dataset_spec in dataset_specs:
        dataset_id, count = parse_dataset_with_count(dataset_spec)

        if count is not None:
            print(f"  Loading {dataset_id} (first {count} samples)...")
        else:
            print(f"  Loading {dataset_id}...")

        dataset = load_dataset(dataset_id)
        ds = dataset[split]

        if count is not None:
            ds = ds.select(range(min(count, len(ds))))
            print(f"    Selected {len(ds)} samples")

        datasets.append(ds)

    if len(datasets) > 1:
        total_samples = sum(len(ds) for ds in datasets)
        print(
            f"Concatenating {len(datasets)} datasets (total: {total_samples} samples)..."
        )
        return concatenate_datasets(datasets)
    else:
        return datasets[0]
