#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


def load_gold_and_pred(
    test_csv: Path, pred_csv: Path
) -> Tuple[list[str], list[str]]:
    if not test_csv.exists():
        raise FileNotFoundError(f"Gold/test CSV not found: {test_csv}")
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    gold: list[str] = []
    with test_csv.open(newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)
        for row in r:
            # Expect 2 columns: sentence, gold constraint
            if len(row) < 2:
                continue
            gold.append(row[1])

    pred: list[str] = []
    with pred_csv.open(newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)
        for row in r:
            # Expect 2 columns: sentence, predicted (raw)
            if len(row) < 2:
                continue
            pred.append(row[1])

    # Align lengths defensively
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

    per_templ_counts: Dict[str, Tuple[int, int]] = {
        k: (per_templ_correct.get(k, 0), per_templ_total[k]) for k in per_templ_total
    }
    return per_templ_acc, overall_acc, per_templ_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce external validation Part A template accuracy from archived predictions"
    )
    parser.add_argument(
        "--dataset",
        choices=["V1", "V2"],
        default="V1",
        help="Which external dataset to evaluate (V1 or V2)",
    )
    parser.add_argument(
        "--repo_root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root (defaults to two levels up)",
    )
    parser.add_argument(
        "--test_csv",
        default=None,
        help="Optional override path for test CSV (gold)",
    )
    parser.add_argument(
        "--pred_csv",
        default=None,
        help="Optional override path for predictions CSV (our LLM outputs)",
    )

    args = parser.parse_args()
    repo_root = Path(args.repo_root)

    # Defaults per dataset
    if args.test_csv is None:
        test_csv = repo_root / "data/external_validation/part_a" / (
            f"test_set_{args.dataset}.csv"
        )
    else:
        test_csv = Path(args.test_csv)

    if args.pred_csv is None:
        pred_csv = repo_root / "data/external_validation/part_a/results" / (
            f"results_{args.dataset}.csv"
        )
    else:
        pred_csv = Path(args.pred_csv)

    gold, pred = load_gold_and_pred(test_csv, pred_csv)
    per_templ_acc, overall_acc, per_templ_counts = compute_template_accuracy(gold, pred)

    print(f"Dataset: {args.dataset}")
    print(f"Gold: {test_csv}")
    print(f"Pred: {pred_csv}")
    print("")
    print("Per-template accuracy (correct/total | accuracy):")
    for templ in sorted(per_templ_acc.keys()):
        corr, tot = per_templ_counts[templ]
        print(f"- {templ:>18}: {corr:3d}/{tot:<3d} | {per_templ_acc[templ]:.3f}")
    print("")
    print(f"Overall template accuracy: {overall_acc:.3f}")


if __name__ == "__main__":
    main()
    # to run: python external_part_a_eval.py --dataset V1


