#!/usr/bin/env python3
"""
Parquet to CSV Converter

Simple utility to convert Parquet files to CSV format.

Usage:
    python parquet_to_csv.py input.parquet
    python parquet_to_csv.py input.parquet --output output.csv
    python parquet_to_csv.py /path/to/folder  # Convert all parquet files in folder
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas pyarrow")
    sys.exit(1)


def convert_parquet_to_csv(parquet_path: str, csv_path: str = None, verbose: bool = True) -> str:
    """
    Convert a Parquet file to CSV.
    
    Args:
        parquet_path: Path to input Parquet file
        csv_path: Path to output CSV file (optional, auto-generated if None)
        verbose: Print progress messages
    
    Returns:
        Path to the created CSV file
    """
    parquet_path = Path(parquet_path)
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    if csv_path is None:
        csv_path = parquet_path.with_suffix('.csv')
    else:
        csv_path = Path(csv_path)
    
    if verbose:
        print(f"Reading: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    
    if verbose:
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Writing: {csv_path}")
    
    df.to_csv(csv_path, index=False)
    
    if verbose:
        # Show file sizes
        parquet_size = parquet_path.stat().st_size / (1024 * 1024)
        csv_size = csv_path.stat().st_size / (1024 * 1024)
        print(f"  Parquet size: {parquet_size:.2f} MB")
        print(f"  CSV size: {csv_size:.2f} MB")
        print(f"  Ratio: {csv_size / parquet_size:.1f}x larger")
    
    return str(csv_path)


def convert_folder(folder_path: str, verbose: bool = True) -> list:
    """
    Convert all Parquet files in a folder to CSV.
    
    Args:
        folder_path: Path to folder containing Parquet files
        verbose: Print progress messages
    
    Returns:
        List of created CSV file paths
    """
    folder_path = Path(folder_path)
    
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")
    
    parquet_files = list(folder_path.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No Parquet files found in: {folder_path}")
        return []
    
    if verbose:
        print(f"Found {len(parquet_files)} Parquet files in: {folder_path}")
        print("-" * 60)
    
    csv_files = []
    for pq_file in parquet_files:
        try:
            csv_path = convert_parquet_to_csv(pq_file, verbose=verbose)
            csv_files.append(csv_path)
            if verbose:
                print()
        except Exception as e:
            print(f"Error converting {pq_file}: {e}")
    
    return csv_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert Parquet files to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert a single file
    python parquet_to_csv.py data.parquet
    
    # Convert with custom output path
    python parquet_to_csv.py data.parquet --output results.csv
    
    # Convert all Parquet files in a folder
    python parquet_to_csv.py /path/to/folder
    
    # Quiet mode
    python parquet_to_csv.py data.parquet --quiet
        """
    )
    parser.add_argument(
        "input", 
        help="Path to Parquet file or folder containing Parquet files"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output CSV path (only for single file conversion)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    verbose = not args.quiet
    
    if input_path.is_dir():
        if args.output:
            print("Warning: --output is ignored when converting a folder")
        csv_files = convert_folder(input_path, verbose=verbose)
        if verbose:
            print("-" * 60)
            print(f"Converted {len(csv_files)} files")
    elif input_path.is_file():
        convert_parquet_to_csv(input_path, args.output, verbose=verbose)
    else:
        print(f"Error: Path does not exist: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
