from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
from datasets import Dataset

from .prompts import PROMPT_QA


@dataclass
class DataColumns:
    text_col: Optional[str] = None
    label_col: Optional[str] = None


def detect_columns(df: pd.DataFrame, text_override: Optional[str], label_override: Optional[str]) -> tuple[str, str]:
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
        raise ValueError("CSV must contain 'Text Description' and 'Declare Constraint' (or provide overrides)")
    return text_col, label_col


def read_training_csv(
    csv_path: str,
    delimiter: Optional[str] = None,
    text_override: Optional[str] = None,
    label_override: Optional[str] = None,
) -> Dataset:
    if delimiter is not None:
        df = pd.read_csv(csv_path, delimiter=delimiter)
    else:
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python")
        except Exception:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                df = pd.read_csv(csv_path, delimiter=";")

    text_col, label_col = detect_columns(df, text_override, label_override)

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["prompt"] = [PROMPT_QA.format(description=t, constraint=l) for t, l in zip(df["text"], df["label"])]
    return Dataset.from_pandas(df[["prompt"]])
