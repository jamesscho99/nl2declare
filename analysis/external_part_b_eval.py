#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple
import sys

# Make src importable when running from repo
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from text2declare.evaluation import evaluate as eval_metrics


def load_gold_and_pred(test_csv: Path, pred_csv: Path) -> Tuple[list[str], list[str]]:
    if not test_csv.exists():
        raise FileNotFoundError(f"Gold/test CSV not found: {test_csv}")
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    gold: list[str] = []
    with test_csv.open(newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            gold.append(row[1])

    pred: list[str] = []
    with pred_csv.open(newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            pred.append(row[1])

    n = min(len(gold), len(pred))
    return gold[:n], pred[:n]


def compute_template_accuracy(gold: list[str], pred: list[str]) -> Tuple[Dict[str, float], float, Dict[str, Tuple[int, int]]]:
    per_templ_total: Dict[str, int] = defaultdict(int)
    per_templ_correct: Dict[str, int] = defaultdict(int)

    for g, p in zip(gold, pred):
        templ_g = g.split("(")[0].strip() if g else ""
        templ_p = p.split("(")[0].strip() if p else ""
        if not templ_g:
            continue
        per_templ_total[templ_g] += 1
        if templ_g == templ_p:
            per_templ_correct[templ_g] += 1

    per_templ_acc: Dict[str, float] = {}
    for k in per_templ_total:
        total_k = per_templ_total[k]
        correct_k = per_templ_correct.get(k, 0)
        per_templ_acc[k] = (correct_k / total_k) if total_k else 0.0

    overall_total = sum(per_templ_total.values())
    overall_correct = sum(per_templ_correct.values())
    overall_acc = (overall_correct / overall_total) if overall_total else 0.0

    per_templ_counts: Dict[str, Tuple[int, int]] = {k: (per_templ_correct.get(k, 0), per_templ_total[k]) for k in per_templ_total}
    return per_templ_acc, overall_acc, per_templ_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce external validation Part B template accuracy from archived predictions")
    parser.add_argument(
        "--repo_root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root (defaults to two levels up)",
    )
    parser.add_argument("--test_csv", default=None, help="Optional override path for test CSV (gold)")
    parser.add_argument("--pred_csv", default=None, help="Override predictions CSV for proposed approach")
    parser.add_argument("--few_shot_csv", default=None, help="Override predictions CSV for few-shot baseline")
    parser.add_argument("--few_shot_cot_csv", default=None, help="Override predictions CSV for few-shot-CoT baseline")

    args = parser.parse_args()
    repo_root = Path(args.repo_root)

    if args.test_csv is None:
        # Preferred location; fallback to original baselines path if not present
        preferred = repo_root / "data/external_validation/part_b/test_data.csv"
        fallback = repo_root / "data/external_validation/baselines_prompting/test_data.csv"
        test_csv = preferred if preferred.exists() else fallback
    else:
        test_csv = Path(args.test_csv)

    # Resolve prediction files
    proposed_csv = Path(args.pred_csv) if args.pred_csv else (repo_root / "data/external_validation/part_b/results/results_test_data_LLM.csv")
    few_shot_csv = Path(args.few_shot_csv) if args.few_shot_csv else (repo_root / "data/external_validation/part_b/results/results_test_data_few_shot.csv")
    few_shot_cot_csv = Path(args.few_shot_cot_csv) if args.few_shot_cot_csv else (repo_root / "data/external_validation/part_b/results/results_test_data_few_shot_cot.csv")

    # Compute metrics using new evaluation framework
    def metrics_for(pred_csv: Path) -> Tuple[float, float, float]:
        template_acc, act_acc, constraint_score, n, te = eval_metrics(str(test_csv), str(pred_csv))
        return template_acc, act_acc, constraint_score

    print(f"Gold: {test_csv}")
    rows = []
    for name, path in (
        ("proposed_LLM", proposed_csv),
        ("few_shot", few_shot_csv),
        ("few_shot_cot", few_shot_cot_csv),
    ):
        if not path.exists():
            rows.append((name, "N/A", "N/A", "N/A", str(path)))
            continue
        template_acc, act_acc, constraint_score = metrics_for(path)
        rows.append((name, f"{template_acc:.3f}", f"{act_acc:.3f}", f"{constraint_score:.3f}", str(path)))

    print("Approach                 TemplateAcc   ActAcc   ConstraintScore    File")
    print("-------------------------------------------------------------------------")
    for name, tacc, aacc, cscore, path in rows:
        print(f"{name:20s}  {tacc:>11s}   {aacc:>6s}   {cscore:>13s}    {path}")


if __name__ == "__main__":
    main()


