from __future__ import annotations
import re
from typing import Optional

# Simple parser for extracting Declare constraint from model output.
# Assumes the output either already in k(A) or contains one such pattern.

CONSTRAINT_PATTERN = re.compile(r"([A-Za-z]+)\s*\(\s*([A-Za-z][A-Za-z\s]*)\s*(?:,\s*([A-Za-z][A-Za-z\s]*))?\s*\)")


def extract_first_constraint(text: str) -> Optional[str]:
    m = CONSTRAINT_PATTERN.search(text)
    if not m:
        return None
    if m.group(3) is None:
        return f"{m.group(1)}({m.group(2).strip()})"
    return f"{m.group(1)}({m.group(2).strip()}, {m.group(3).strip()})"
