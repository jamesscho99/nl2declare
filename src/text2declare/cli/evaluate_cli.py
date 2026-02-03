from __future__ import annotations
from text2declare.evaluation import evaluate as _evaluate


def run_cli(ground_truth_file: str, found_file: str, alpha: float = 2.0) -> None:
    precision, recall, f1, num_samples, template_errors = _evaluate(ground_truth_file, found_file, alpha)
    print(f"Overall precision: {precision}")
    print(f"Overall recall: {recall}")
    print(f"Overall F1: {f1}")
    print(f"No. of sentences/constraints in test dataset: {num_samples}")
    print(f"No. of template errors: {template_errors}")
    print(f"Template accuracy: {1 - (template_errors / num_samples if num_samples else 0.0)}")
