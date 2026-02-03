import argparse
from .evaluate_cli import run_cli


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted Declare constraints against ground truth.")
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--alpha", type=float, default=2.0)
    args = parser.parse_args()
    run_cli(args.ground_truth, args.predictions, args.alpha)
