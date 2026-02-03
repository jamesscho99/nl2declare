import argparse
import os
from dataclasses import dataclass
from typing import List

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer


PROMPT_QA = (
    "### Question: What is the Declare constraint described in the following sentence? {description}\n"
    "### Answer: {constraint}"
)


def load_and_format(csv_path: str, delimiter: str | None = None, text_override: str | None = None, label_override: str | None = None) -> Dataset:
    # Read with robust delimiter handling: try user-provided, else infer, else fallbacks
    if delimiter is not None:
        df = pd.read_csv(csv_path, delimiter=delimiter)
    else:
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python")  # auto-detect
        except Exception:
            # Fallbacks: comma, then semicolon
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                df = pd.read_csv(csv_path, delimiter=';')

    # Expect columns: 'Text Description' and 'Declare Constraint'
    # Allow overrides via CLI
    if text_override is not None and text_override in df.columns:
        text_col = text_override
    else:
        text_col = None
        for c in df.columns:
            if c.strip().strip('"').lower() in {"text description", "text", "sentence"}:
                text_col = c
                break

    if label_override is not None and label_override in df.columns:
        label_col = label_override
    else:
        label_col = None
        for c in df.columns:
            if c.strip().strip('"').lower() in {"declare constraint", "constraint", "label"}:
                label_col = c
                break

    if text_col is None or label_col is None:
        raise ValueError("CSV must contain 'Text Description' and 'Declare Constraint' (or provide --text_col/--label_col)")

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["prompt"] = [PROMPT_QA.format(description=t, constraint=l) for t, l in zip(df["text"], df["label"])]
    return Dataset.from_pandas(df[["prompt"]])


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for NL->Declare with parameterized model")
    parser.add_argument("--model_id", required=True, help="Base model id, e.g., meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--train_csv", required=True, help="Path to CSV with 'Text Description' and 'Declare Constraint'")
    parser.add_argument("--text_col", default=None, help="Override text column name if headers differ")
    parser.add_argument("--label_col", default=None, help="Override label column name if headers differ")
    parser.add_argument("--output_dir", required=True, help="Where to save outputs")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""))
    # Training args
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=None)
    # LoRA
    parser.add_argument("--lora_r", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_and_format(
        args.train_csv,
        delimiter=None,
        text_override=args.text_col,
        label_override=args.label_col,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=args.hf_token,
        torch_dtype=dtype,
        device_map=device_map,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        bf16=False,
        fp16=False,
        logging_steps=10,
        save_steps=args.max_steps,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="prompt",
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    trainer.train()
    # Save adapter weights
    trainer.model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()


