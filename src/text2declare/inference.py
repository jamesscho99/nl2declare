from __future__ import annotations
from typing import Iterable, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .prompts import FEW_SHOT_PROMPT, FEW_SHOT_COT_PROMPT_TEMPLATE


def load_model(model_id: str, hf_token: str = ""):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=hf_token, torch_dtype=torch.float16, device_map={"": 0}
    )
    return model, tokenizer


def load_peft_model(model_id: str, adapter_dir: str, hf_token: str = ""):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=hf_token, torch_dtype=torch.float16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def predict_constraints(model, tokenizer, sentences: Iterable[str], style: str = "few_shot", max_new_tokens: int = 64) -> List[str]:
    outputs: List[str] = []
    for s in sentences:
        if style == "few_shot":
            prompt = FEW_SHOT_PROMPT.format(description=s)
        else:
            prompt = FEW_SHOT_COT_PROMPT_TEMPLATE.format(description=s)
        outputs.append(generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens))
    return outputs


def build_qa_prompt(description: str) -> str:
    # Use the SFT QA format: provide question and leave answer empty for the model to complete
    return (
        "### Question: What is the Declare constraint described in the following sentence? "
        f"{description}\n### Answer:"
    )


def predict_constraints_qa(model, tokenizer, sentences: Iterable[str], max_new_tokens: int = 15) -> List[str]:
    outputs: List[str] = []
    for s in sentences:
        prompt = build_qa_prompt(s)
        outputs.append(generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens))
    return outputs
