#!/usr/bin/env python3
"""
Internal validation evaluation script for Table tab:LLM_comparison in main.tex.
This script processes 10-fold cross-validation results for different LLM models
and computes the direction-sensitive metrics: TemplateAcc, ActAcc (conditional), and ConstraintScore.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

# Add the src directory to the path to import evaluation module
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from text2declare.evaluation import evaluate_detailed


def get_model_directories(data_root: Path) -> Dict[str, Path]:
    """
    Get the mapping of model names to their result directories.
    
    Returns:
        Dictionary mapping model display names to their data directories
    """
    model_mapping = {
        "Gemma 7B (ft)": data_root / "Gemma_7B",
        "Mistral 7B (ft)": data_root / "Mistral", 
        "Llama-3 8B (ft)": data_root / "Llama-3",
        "Falcon 7B (ft)": data_root / "Falcon",
        "Llama-2 7B (ft)": data_root / "Llama-2",
        "Llama-3.2 3B-Instruct": data_root / "llama-3B Instruct" / "llama-3B Instruct" / "crossvalidation",
        "Llama-3.2 1B-Instruct": data_root / "llama-1B Instruct" / "llama-1B Instruct" / "crossvalidation"
    }
    
    # Filter to only existing directories
    existing_models = {}
    for model_name, model_dir in model_mapping.items():
        if model_dir.exists():
            existing_models[model_name] = model_dir
        else:
            print(f"Warning: Directory not found for {model_name}: {model_dir}")
    
    return existing_models


def evaluate_model_fold(model_dir: Path, fold_num: int) -> Tuple[float, float, float]:
    """
    Evaluate a single fold for a model.
    
    Args:
        model_dir: Path to the model's result directory
        fold_num: Fold number (1-10)
        
    Returns:
        Tuple of (template_acc, act_acc_conditional, constraint_score)
    """
    test_file = model_dir / f"test_fold_{fold_num}.csv"
    results_file = model_dir / f"results_test_fold_{fold_num}.csv"
    
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    # Use the existing evaluation function (files are now comma-delimited)
    metrics = evaluate_detailed(str(test_file), str(results_file))
    
    return (
        metrics['template_acc'],
        metrics['act_acc_conditional'], 
        metrics['constraint_score']
    )


def evaluate_model_crossvalidation(model_dir: Path, model_name: str) -> Tuple[float, float, float, List[float], List[float], List[float]]:
    """
    Evaluate all 10 folds for a model and compute average metrics.
    
    Args:
        model_dir: Path to the model's result directory
        model_name: Name of the model for error reporting
        
    Returns:
        Tuple of (avg_template_acc, avg_act_acc, avg_constraint_score, 
                 template_acc_scores, act_acc_scores, constraint_scores)
    """
    template_acc_scores = []
    act_acc_scores = []
    constraint_scores = []
    
    for fold in range(1, 11):  # Folds 1-10
        try:
            template_acc, act_acc, constraint_score = evaluate_model_fold(model_dir, fold)
            template_acc_scores.append(template_acc)
            act_acc_scores.append(act_acc)
            constraint_scores.append(constraint_score)
        except FileNotFoundError as e:
            print(f"Warning: Missing file for {model_name} fold {fold}: {e}")
            continue
    
    if not template_acc_scores:
        raise ValueError(f"No valid folds found for model {model_name}")
    
    # Compute averages
    avg_template_acc = statistics.mean(template_acc_scores)
    avg_act_acc = statistics.mean(act_acc_scores)
    avg_constraint_score = statistics.mean(constraint_scores)
    
    return (avg_template_acc, avg_act_acc, avg_constraint_score, 
            template_acc_scores, act_acc_scores, constraint_scores)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce internal validation Table tab:LLM_comparison from main.tex"
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Path to internal validation data directory (defaults to nl2declare/data/internal_validation)"
    )
    parser.add_argument(
        "--repo_root", 
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root (defaults to one level up)"
    )
    parser.add_argument(
        "--show_details",
        action="store_true",
        help="Show detailed fold-by-fold results for each model"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Specific models to evaluate (default: all available models)"
    )
    
    args = parser.parse_args()
    
    # Set up paths
    repo_root = Path(args.repo_root)
    if args.data_root is None:
        data_root = repo_root / "data" / "internal_validation"
    else:
        data_root = Path(args.data_root)
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    # Get available models
    all_models = get_model_directories(data_root)
    
    if args.models:
        # Filter to requested models
        requested_models = {}
        for model_name in args.models:
            if model_name in all_models:
                requested_models[model_name] = all_models[model_name]
            else:
                print(f"Warning: Model '{model_name}' not found. Available models: {list(all_models.keys())}")
        models_to_evaluate = requested_models
    else:
        models_to_evaluate = all_models
    
    if not models_to_evaluate:
        print("No models to evaluate!")
        return
    
    print("=== Internal Validation Results (10-fold Cross-Validation) ===")
    print(f"Data directory: {data_root}")
    print("")
    
    # Store results for table formatting
    results = {}
    
    # Evaluate each model
    for model_name, model_dir in models_to_evaluate.items():
        try:
            print(f"Evaluating {model_name}...")
            avg_template_acc, avg_act_acc, avg_constraint_score, template_scores, act_scores, constraint_scores = evaluate_model_crossvalidation(
                model_dir, model_name
            )
            
            results[model_name] = {
                'template_acc': avg_template_acc,
                'act_acc': avg_act_acc, 
                'constraint_score': avg_constraint_score,
                'template_scores': template_scores,
                'act_scores': act_scores,
                'constraint_scores': constraint_scores
            }
            
            if args.show_details:
                print(f"  Fold-by-fold TemplateAcc: {[f'{s:.3f}' for s in template_scores]}")
                print(f"  Fold-by-fold ActAcc:      {[f'{s:.3f}' for s in act_scores]}")
                print(f"  Fold-by-fold ConstraintScore: {[f'{s:.3f}' for s in constraint_scores]}")
                print(f"  Averages: TemplateAcc={avg_template_acc:.3f}, ActAcc={avg_act_acc:.3f}, ConstraintScore={avg_constraint_score:.3f}")
                print("")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue
    
    if not results:
        print("No successful evaluations!")
        return
    
    # Print summary table (matching Table tab:LLM_comparison format)
    print("\n=== Summary Table (Direction-Sensitive Metrics) ===")
    print(f"{'Model':<25} {'TemplateAcc':<12} {'ActAcc (cond.)':<15} {'ConstraintScore':<15}")
    print("-" * 70)
    
    # Sort by ConstraintScore (descending) to match paper ordering
    sorted_results = sorted(results.items(), key=lambda x: x[1]['constraint_score'], reverse=True)
    
    for model_name, metrics in sorted_results:
        template_acc = metrics['template_acc']
        act_acc = metrics['act_acc']
        constraint_score = metrics['constraint_score']
        
        # Mark the best score in each column with **bold** formatting
        template_str = f"{template_acc:.3f}"
        act_str = f"{act_acc:.3f}"
        constraint_str = f"{constraint_score:.3f}"
        
        # Find best scores
        best_template = max(results.values(), key=lambda x: x['template_acc'])['template_acc']
        best_act = max(results.values(), key=lambda x: x['act_acc'])['act_acc']
        best_constraint = max(results.values(), key=lambda x: x['constraint_score'])['constraint_score']
        
        if abs(template_acc - best_template) < 1e-6:
            template_str = f"**{template_str}**"
        if abs(act_acc - best_act) < 1e-6:
            act_str = f"**{act_str}**"
        if abs(constraint_score - best_constraint) < 1e-6:
            constraint_str = f"**{constraint_str}**"
            
        print(f"{model_name:<25} {template_str:<12} {act_str:<15} {constraint_str:<15}")
    
    print("")
    print("Note: Values marked with ** indicate the best performance in each metric.")
    print("These results correspond to Table tab:LLM_comparison in main.tex.")


if __name__ == "__main__":
    # # Run all models
    # python analysis/internal_validation_eval.py

    # # Run specific models with detailed output  
    # python analysis/internal_validation_eval.py --models "Gemma 7B (ft)" --show_details

    # # Use custom data directory
    # python analysis/internal_validation_eval.py --data_root /path/to/data
    main()
