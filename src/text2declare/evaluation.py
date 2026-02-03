import csv
from collections import defaultdict

def calculate_new_metrics(ground_truth, found):
    """
    Calculate the new metrics as specified in main.tex:
    - TemplateAcc_i: whether template is correct
    - ActAcc_i: activity accuracy (only if template correct and arity matches)
    - ConstraintScore_i: product of template and activity accuracy
    
    Parameters:
    - ground_truth: Dictionary of ground-truth constraints.
    - found: Dictionary of found constraints.
    
    Returns:
    - List of tuples (template_acc, act_acc, constraint_score) for each instance
    - Dictionary of per-template scores for macro-averaging
    """
    scores = []
    per_template_scores = defaultdict(list)
    template_errors = 0
    num_samples = 0

    # Define symmetric templates (where argument order doesn't matter)
    symmetric_templates = {'CoExistence', 'NotCoExistence'}

    for sentence, ground_constraint in ground_truth.items():
        num_samples += 1
        template_acc = 0.0
        act_acc = 0.0
        constraint_score = 0.0
        
        if sentence not in found:
            # No prediction found
            scores.append((template_acc, act_acc, constraint_score))
            template_errors += 1
            continue
            
        found_constraint = found[sentence]
        if not found_constraint or found_constraint == "":
            # Empty prediction
            scores.append((template_acc, act_acc, constraint_score))
            template_errors += 1
            continue

        # Parse ground truth constraint
        try:
            g_template = ground_constraint.split('(')[0].strip()
            g_args_raw = ground_constraint.split('(')[1].rstrip(')')
            g_args = [a.strip() for a in g_args_raw.split(',') if a.strip()]
            m = len(g_args)  # number of activities in ground truth
        except Exception:
            # Malformed ground truth
            scores.append((template_acc, act_acc, constraint_score))
            continue

        # Parse found constraint
        if '(' in found_constraint and ')' in found_constraint:
            try:
                f_template = found_constraint.split('(')[0].strip()
                f_args_raw = found_constraint.split('(')[1].rstrip(')')
                f_args = [a.strip() for a in f_args_raw.split(',') if a.strip()]
                n = len(f_args)  # number of activities in prediction
            except Exception:
                # Malformed prediction
                f_template = None
                f_args = []
                n = 0
        else:
            f_template = None
            f_args = []
            n = 0

        # Calculate TemplateAcc_i
        if g_template == f_template:
            template_acc = 1.0
        else:
            template_acc = 0.0
            template_errors += 1

        # Calculate ActAcc_i (only if template correct and arity matches)
        if template_acc == 1.0 and n == m:
            if m == 0:
                act_acc = 1.0  # No activities to check
            else:
                # Direction-sensitive matching by default
                correct_matches = 0
                for j in range(m):
                    if j < len(f_args) and g_args[j] == f_args[j]:
                        correct_matches += 1
                
                # For symmetric templates, also try swapped order if binary
                if g_template in symmetric_templates and m == 2:
                    # Try swapped order and take maximum
                    swapped_matches = 0
                    if g_args[0] == f_args[1] and g_args[1] == f_args[0]:
                        swapped_matches = 2
                    correct_matches = max(correct_matches, swapped_matches)
                
                act_acc = correct_matches / m
        else:
            act_acc = 0.0

        # Calculate ConstraintScore_i
        constraint_score = template_acc * act_acc

        scores.append((template_acc, act_acc, constraint_score))
        
        # Store for per-template macro-averaging
        per_template_scores[g_template].append((template_acc, act_acc, constraint_score))

    return scores, per_template_scores, num_samples, template_errors

def compute_macro_averaged_metrics(scores, per_template_scores):
    """
    Compute macro-averaged metrics as specified in main.tex:
    - TemplateAcc: macro-average over all instances
    - ActAcc: conditional on correct template (only instances with correct template)
    - ConstraintScore: macro-average over all instances
    - Per-template macro-averages
    
    Parameters:
    - scores: List of tuples (template_acc, act_acc, constraint_score) for each instance
    - per_template_scores: Dictionary mapping template names to lists of scores
    
    Returns:
    - Dictionary with overall and per-template metrics
    """
    N = len(scores)
    
    if N == 0:
        return {
            'template_acc': 0.0,
            'act_acc_conditional': 0.0,
            'constraint_score': 0.0,
            'per_template': {}
        }
    
    # Overall metrics
    template_acc_sum = sum(score[0] for score in scores)
    act_acc_sum = sum(score[1] for score in scores if score[0] == 1.0)  # Only correct templates
    constraint_score_sum = sum(score[2] for score in scores)
    
    # Count instances with correct templates for conditional ActAcc
    correct_template_count = sum(1 for score in scores if score[0] == 1.0)
    
    template_acc = template_acc_sum / N
    act_acc_conditional = act_acc_sum / max(1, correct_template_count)  # Avoid division by zero
    constraint_score = constraint_score_sum / N
    
    # Per-template macro-averages
    per_template_metrics = {}
    for template_name, template_scores in per_template_scores.items():
        if len(template_scores) > 0:
            template_template_acc = sum(score[0] for score in template_scores) / len(template_scores)
            template_correct_count = sum(1 for score in template_scores if score[0] == 1.0)
            template_act_acc = (sum(score[1] for score in template_scores if score[0] == 1.0) / 
                               max(1, template_correct_count))
            template_constraint_score = sum(score[2] for score in template_scores) / len(template_scores)
            
            per_template_metrics[template_name] = {
                'template_acc': template_template_acc,
                'act_acc_conditional': template_act_acc,
                'constraint_score': template_constraint_score,
                'count': len(template_scores)
            }
    
    return {
        'template_acc': template_acc,
        'act_acc_conditional': act_acc_conditional,
        'constraint_score': constraint_score,
        'per_template': per_template_metrics
    }

def evaluate(ground_truth_file, found_file, alpha: float = 2.0):
    """
    Evaluate the constraints using the new metrics from main.tex.
    
    Parameters:
    - ground_truth_file: Path to the ground-truth CSV file.
    - found_file: Path to the found constraints CSV file.
    - alpha: Legacy parameter (kept for backward compatibility, not used in new metrics)
    
    Returns:
    - Tuple (template_acc, act_acc_conditional, constraint_score, num_samples, template_errors)
    """
    ground_truth = {}
    with open(ground_truth_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                ground_truth[row[0]] = row[1]

    found = {}
    with open(found_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                found[row[0]] = row[1]

    scores, per_template_scores, num_samples, template_errors = calculate_new_metrics(ground_truth, found)
    metrics = compute_macro_averaged_metrics(scores, per_template_scores)
    
    return (
        metrics['template_acc'],
        metrics['act_acc_conditional'], 
        metrics['constraint_score'],
        num_samples,
        template_errors
    )

def evaluate_detailed(ground_truth_file, found_file):
    """
    Evaluate constraints and return detailed metrics including per-template breakdown.
    
    Parameters:
    - ground_truth_file: Path to the ground-truth CSV file.
    - found_file: Path to the found constraints CSV file.
    
    Returns:
    - Dictionary with detailed metrics
    """
    ground_truth = {}
    with open(ground_truth_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                ground_truth[row[0]] = row[1]

    found = {}
    with open(found_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                found[row[0]] = row[1]

    scores, per_template_scores, num_samples, template_errors = calculate_new_metrics(ground_truth, found)
    metrics = compute_macro_averaged_metrics(scores, per_template_scores)
    
    # Add additional summary information
    metrics['num_samples'] = num_samples
    metrics['template_errors'] = template_errors
    
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate constraints CSVs using new metrics from main.tex")
    parser.add_argument("--ground_truth", required=True, help="Path to ground truth CSV file")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV file")
    parser.add_argument("--alpha", type=float, default=2.0, help="Legacy parameter (not used in new metrics)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed per-template breakdown")
    args = parser.parse_args()
    
    if args.detailed:
        metrics = evaluate_detailed(args.ground_truth, args.predictions)
        print("=== Detailed Evaluation Results ===")
        print(f"Template Accuracy: {metrics['template_acc']:.3f}")
        print(f"Activity Accuracy (conditional): {metrics['act_acc_conditional']:.3f}")
        print(f"Constraint Score: {metrics['constraint_score']:.3f}")
        print(f"No. of sentences/constraints: {metrics['num_samples']}")
        print(f"No. of template errors: {metrics['template_errors']}")
        
        print("\n=== Per-Template Breakdown ===")
        for template, scores in metrics['per_template'].items():
            print(f"{template} ({scores['count']} instances):")
            print(f"  Template Acc: {scores['template_acc']:.3f}")
            print(f"  Activity Acc: {scores['act_acc_conditional']:.3f}")
            print(f"  Constraint Score: {scores['constraint_score']:.3f}")
    else:
        template_acc, act_acc, constraint_score, num_samples, template_errors = evaluate(
            args.ground_truth, args.predictions, args.alpha
        )
        print("=== Evaluation Results ===")
        print(f"Template Accuracy: {template_acc:.3f}")
        print(f"Activity Accuracy (conditional): {act_acc:.3f}")
        print(f"Constraint Score: {constraint_score:.3f}")
        print(f"No. of sentences/constraints: {num_samples}")
        print(f"No. of template errors: {template_errors}")
        
        # For backward compatibility, also show legacy template accuracy calculation
        legacy_template_acc = 1 - (template_errors / num_samples if num_samples else 0.0)
        print(f"Legacy template accuracy: {legacy_template_acc:.3f}")