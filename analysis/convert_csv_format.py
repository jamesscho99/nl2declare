#!/usr/bin/env python3
"""
Convert all semicolon-delimited CSV files in the internal validation data
to comma-delimited format and save them in place.
"""

import argparse
import csv
from pathlib import Path
from typing import List


def convert_semicolon_to_comma_csv(input_file: Path) -> None:
    """
    Convert a semicolon-delimited CSV file to comma-delimited format in place.
    
    Args:
        input_file: Path to the CSV file to convert
    """
    # Read the original file
    rows = []
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=';')
        for row in reader:
            rows.append(row)
    
    # Write back as comma-delimited
    with open(input_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        for row in rows:
            writer.writerow(row)


def find_csv_files(data_root: Path) -> List[Path]:
    """
    Find all CSV files in the internal validation directory structure.
    
    Args:
        data_root: Root directory for internal validation data
        
    Returns:
        List of paths to CSV files
    """
    csv_files = []
    
    # Get all CSV files recursively
    for csv_file in data_root.rglob("*.csv"):
        # Skip the main crossvalidation data file
        if csv_file.name == "data_crossvalidation.csv":
            continue
        csv_files.append(csv_file)
    
    return sorted(csv_files)


def main():
    parser = argparse.ArgumentParser(
        description="Convert semicolon-delimited CSV files to comma-delimited format"
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
        "--dry_run",
        action="store_true",
        help="Show what would be converted without actually doing it"
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
    
    # Find all CSV files
    csv_files = find_csv_files(data_root)
    
    print(f"Found {len(csv_files)} CSV files to convert in {data_root}")
    
    if args.dry_run:
        print("\nFiles that would be converted (dry run):")
        for csv_file in csv_files:
            rel_path = csv_file.relative_to(data_root)
            print(f"  {rel_path}")
        print(f"\nTotal: {len(csv_files)} files")
        return
    
    # Convert files
    converted_count = 0
    error_count = 0
    
    for csv_file in csv_files:
        try:
            rel_path = csv_file.relative_to(data_root)
            print(f"Converting {rel_path}...")
            convert_semicolon_to_comma_csv(csv_file)
            converted_count += 1
        except Exception as e:
            print(f"Error converting {csv_file}: {e}")
            error_count += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {converted_count} files")
    if error_count > 0:
        print(f"Errors: {error_count} files")
    
    print(f"\nAll CSV files in {data_root} are now comma-delimited.")


if __name__ == "__main__":
    main()
