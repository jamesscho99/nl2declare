from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from .data import read_training_csv


@dataclass
class LoraArgs:
    r: int = 8


@dataclass
class TrainConfig:
    model_id: str
    train_csv: str
    output_dir: str
    hf_token: str = ""
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 2
    max_steps: int = 200
    learning_rate: float = 2e-4
    seed: Optional[int] = None
    lora: LoraArgs = LoraArgs()


def create_model_and_tokenizer(cfg: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=cfg.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else None
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, token=cfg.hf_token, torch_dtype=dtype, device_map=device_map
    )
    lora_config = LoraConfig(
        r=cfg.lora.r,
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
    return model, tokenizer


def train(cfg: TrainConfig, text_col: Optional[str] = None, label_col: Optional[str] = None):
    dataset = read_training_csv(cfg.train_csv, delimiter=None, text_override=text_col, label_override=label_col)
    model, tokenizer = create_model_and_tokenizer(cfg)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        bf16=False,
        fp16=False,
        logging_steps=10,
        save_steps=cfg.max_steps,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="prompt",
    )

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    trainer.train()
    trainer.model.save_pretrained(cfg.output_dir + "/adapter")
    tokenizer.save_pretrained(cfg.output_dir)
