#!/usr/bin/env python3
"""
H5 Sequence Statistics - Fast parallel statistics for large HDF5 sequence datasets.

Computes detailed statistics about sequence properties:
- How many sequences have a token at index 6 with age in 0-9 (childhood)
- How many sequences have length 1024 (non-zero token or age at index 1023)
- How many sequences have age at index 1023 in decades 70-79, 80-89, 90-99
- Age distributions at index 0 and 1023
- (age_0, age_1023) pair frequencies
- Combined criteria counts

Usage:
    python h5_sequence_statistics.py --h5_file encoded.h5 --output stats_report.txt
    python h5_sequence_statistics.py --h5_file encoded.h5 --n_workers 16

The HDF5 file should have 'input_ids' with shape (N, 4, 1024):
    - input_ids[:, 0, :] = token IDs
    - input_ids[:, 2, :] = ages

Performance optimizations:
    - Vectorized numpy operations (no Python loops over sequences)
    - Large batch processing (default 500K sequences per batch)
    - Multi-process parallel chunk processing
    - Memory-mapped reading for large files
    - tqdm progress bars for tracking
"""

import argparse
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from functools import partial

import h5py
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Decade boundaries for vectorized binning
DECADE_BINS = [-1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
DECADE_LABELS = ["negative", "0-9", "10-19", "20-29", "30-39", "40-49", 
                 "50-59", "60-69", "70-79", "80-89", "90-99", "100+"]


def get_decade(age: int) -> str:
    """Convert age to decade string."""
    if age < 0:
        return "negative"
    elif age < 10:
        return "0-9"
    elif age < 20:
        return "10-19"
    elif age < 30:
        return "20-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    elif age < 70:
        return "60-69"
    elif age < 80:
        return "70-79"
    elif age < 90:
        return "80-89"
    elif age < 100:
        return "90-99"
    else:
        return "100+"


def get_decade_vectorized(ages: np.ndarray) -> np.ndarray:
    """Vectorized conversion of ages to decade indices."""
    # Returns indices into DECADE_LABELS
    return np.digitize(ages, DECADE_BINS[1:])  # 0-11 indices


def process_chunk_vectorized(h5_path: str, start_idx: int, end_idx: int) -> Dict:
    """
    Process a chunk of sequences using vectorized numpy operations.
    
    This is MUCH faster than the loop-based version for large datasets.
    
    Args:
        h5_path: Path to HDF5 file
        start_idx: Start index (inclusive)
        end_idx: End index (exclusive)
    
    Returns:
        Dictionary with statistics for this chunk
    """
    with h5py.File(h5_path, 'r', swmr=True) as f:
        # Read the chunk - only the columns we need
        input_ids = f['input_ids'][start_idx:end_idx]
        
        tokens = input_ids[:, 0, :]  # (chunk_size, 1024)
        ages = input_ids[:, 2, :]    # (chunk_size, 1024)
        
        n_sequences = tokens.shape[0]
        
        # Extract key columns
        age_0 = ages[:, 0].astype(np.int32)
        age_6 = ages[:, 6].astype(np.int32)
        age_1023 = ages[:, 1023].astype(np.int32)
        token_1023 = tokens[:, 1023].astype(np.int32)
        
        # Vectorized criteria checks
        childhood_start = (age_6 >= 0) & (age_6 <= 9)
        full_length = (token_1023 != 0) | (age_1023 != 0)
        last_3_decades = (age_1023 >= 70) & (age_1023 <= 99)
        
        # Count criteria
        stats = {
            'total': n_sequences,
            'childhood_start': int(childhood_start.sum()),
            'full_length': int(full_length.sum()),
            'last_3_decades': int(last_3_decades.sum()),
            'all_criteria': int((childhood_start & full_length & last_3_decades).sum()),
            'childhood_and_full': int((childhood_start & full_length).sum()),
            'childhood_and_last3': int((childhood_start & last_3_decades).sum()),
            'full_and_last3': int((full_length & last_3_decades).sum()),
            'token_1023_zero': int((token_1023 == 0).sum()),
            'token_1023_nonzero': int((token_1023 != 0).sum()),
        }
        
        # Age distributions using np.unique for efficiency
        age_0_unique, age_0_counts = np.unique(age_0, return_counts=True)
        age_6_unique, age_6_counts = np.unique(age_6, return_counts=True)
        age_1023_unique, age_1023_counts = np.unique(age_1023, return_counts=True)
        
        stats['age_at_0'] = Counter(dict(zip(age_0_unique.tolist(), age_0_counts.tolist())))
        stats['age_at_6'] = Counter(dict(zip(age_6_unique.tolist(), age_6_counts.tolist())))
        stats['age_at_1023'] = Counter(dict(zip(age_1023_unique.tolist(), age_1023_counts.tolist())))
        
        # Decade distributions using vectorized binning
        decade_idx_0 = get_decade_vectorized(age_0)
        decade_idx_6 = get_decade_vectorized(age_6)
        decade_idx_1023 = get_decade_vectorized(age_1023)
        
        stats['decade_at_0'] = Counter()
        stats['decade_at_6'] = Counter()
        stats['decade_at_1023'] = Counter()
        
        for idx in range(len(DECADE_LABELS)):
            label = DECADE_LABELS[idx]
            stats['decade_at_0'][label] = int((decade_idx_0 == idx).sum())
            stats['decade_at_6'][label] = int((decade_idx_6 == idx).sum())
            stats['decade_at_1023'][label] = int((decade_idx_1023 == idx).sum())
        
        # Age pair frequencies - use structured array for efficiency
        # Create pair keys as (age_0 * 1000 + age_1023) for fast counting
        pair_keys = age_0.astype(np.int64) * 1000 + age_1023.astype(np.int64)
        pair_unique, pair_counts = np.unique(pair_keys, return_counts=True)
        
        stats['age_pair_0_1023'] = Counter()
        for key, count in zip(pair_unique.tolist(), pair_counts.tolist()):
            a0 = key // 1000
            a1023 = key % 1000
            stats['age_pair_0_1023'][(a0, a1023)] = count
    
    return stats


def merge_stats(stats_list: List[Dict]) -> Dict:
    """Merge statistics from multiple chunks."""
    merged = {
        'total': 0,
        'childhood_start': 0,
        'full_length': 0,
        'last_3_decades': 0,
        'all_criteria': 0,
        'age_at_0': Counter(),
        'age_at_6': Counter(),
        'age_at_1023': Counter(),
        'decade_at_0': Counter(),
        'decade_at_6': Counter(),
        'decade_at_1023': Counter(),
        'age_pair_0_1023': Counter(),
        'token_1023_zero': 0,
        'token_1023_nonzero': 0,
        'childhood_and_full': 0,
        'childhood_and_last3': 0,
        'full_and_last3': 0,
    }
    
    for stats in stats_list:
        merged['total'] += stats['total']
        merged['childhood_start'] += stats['childhood_start']
        merged['full_length'] += stats['full_length']
        merged['last_3_decades'] += stats['last_3_decades']
        merged['all_criteria'] += stats['all_criteria']
        merged['token_1023_zero'] += stats['token_1023_zero']
        merged['token_1023_nonzero'] += stats['token_1023_nonzero']
        merged['childhood_and_full'] += stats['childhood_and_full']
        merged['childhood_and_last3'] += stats['childhood_and_last3']
        merged['full_and_last3'] += stats['full_and_last3']
        
        merged['age_at_0'].update(stats['age_at_0'])
        merged['age_at_6'].update(stats['age_at_6'])
        merged['age_at_1023'].update(stats['age_at_1023'])
        merged['decade_at_0'].update(stats['decade_at_0'])
        merged['decade_at_6'].update(stats['decade_at_6'])
        merged['decade_at_1023'].update(stats['decade_at_1023'])
        merged['age_pair_0_1023'].update(stats['age_pair_0_1023'])
    
    return merged


def format_percentage(count: int, total: int) -> str:
    """Format count as percentage."""
    if total == 0:
        return "0.00%"
    return f"{100 * count / total:.4f}%"


def format_report(stats: Dict) -> str:
    """Format statistics into a human-readable report."""
    total = stats['total']
    
    lines = []
    lines.append("=" * 80)
    lines.append("H5 SEQUENCE STATISTICS REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total sequences:                    {total:,}")
    lines.append("")
    
    # Main criteria
    lines.append("MAIN CRITERIA")
    lines.append("-" * 40)
    lines.append(f"1. Childhood start (age at idx 6 in 0-9): {stats['childhood_start']:,} ({format_percentage(stats['childhood_start'], total)})")
    lines.append(f"2. Full length (non-zero at idx 1023):    {stats['full_length']:,} ({format_percentage(stats['full_length'], total)})")
    lines.append(f"3. Last 3 decades (age 1023 in 70-99):    {stats['last_3_decades']:,} ({format_percentage(stats['last_3_decades'], total)})")
    lines.append("")
    
    # Criteria combinations
    lines.append("CRITERIA COMBINATIONS")
    lines.append("-" * 40)
    lines.append(f"Childhood + Full length:           {stats['childhood_and_full']:,} ({format_percentage(stats['childhood_and_full'], total)})")
    lines.append(f"Childhood + Last 3 decades:        {stats['childhood_and_last3']:,} ({format_percentage(stats['childhood_and_last3'], total)})")
    lines.append(f"Full length + Last 3 decades:      {stats['full_and_last3']:,} ({format_percentage(stats['full_and_last3'], total)})")
    lines.append(f"ALL THREE CRITERIA:                {stats['all_criteria']:,} ({format_percentage(stats['all_criteria'], total)})")
    lines.append("")
    
    # Token at index 1023
    lines.append("TOKEN AT INDEX 1023")
    lines.append("-" * 40)
    lines.append(f"Zero (padding):     {stats['token_1023_zero']:,} ({format_percentage(stats['token_1023_zero'], total)})")
    lines.append(f"Non-zero (real):    {stats['token_1023_nonzero']:,} ({format_percentage(stats['token_1023_nonzero'], total)})")
    lines.append("")
    
    # Decade distributions
    decade_order = ["negative", "0-9", "10-19", "20-29", "30-39", "40-49", 
                    "50-59", "60-69", "70-79", "80-89", "90-99", "100+"]
    
    lines.append("DECADE DISTRIBUTION AT INDEX 0")
    lines.append("-" * 40)
    for decade in decade_order:
        count = stats['decade_at_0'].get(decade, 0)
        if count > 0:
            lines.append(f"  {decade:10s}: {count:>15,} ({format_percentage(count, total)})")
    lines.append("")
    
    lines.append("DECADE DISTRIBUTION AT INDEX 6 (First real token)")
    lines.append("-" * 40)
    for decade in decade_order:
        count = stats['decade_at_6'].get(decade, 0)
        if count > 0:
            lines.append(f"  {decade:10s}: {count:>15,} ({format_percentage(count, total)})")
    lines.append("")
    
    lines.append("DECADE DISTRIBUTION AT INDEX 1023 (Last position)")
    lines.append("-" * 40)
    for decade in decade_order:
        count = stats['decade_at_1023'].get(decade, 0)
        if count > 0:
            lines.append(f"  {decade:10s}: {count:>15,} ({format_percentage(count, total)})")
    lines.append("")
    
    # Individual age distributions (top 20)
    lines.append("AGE DISTRIBUTION AT INDEX 0 (Top 20)")
    lines.append("-" * 40)
    for age, count in stats['age_at_0'].most_common(20):
        lines.append(f"  Age {age:3d}: {count:>15,} ({format_percentage(count, total)})")
    lines.append("")
    
    lines.append("AGE DISTRIBUTION AT INDEX 6 (Top 20)")
    lines.append("-" * 40)
    for age, count in stats['age_at_6'].most_common(20):
        lines.append(f"  Age {age:3d}: {count:>15,} ({format_percentage(count, total)})")
    lines.append("")
    
    lines.append("AGE DISTRIBUTION AT INDEX 1023 (Top 20)")
    lines.append("-" * 40)
    for age, count in stats['age_at_1023'].most_common(20):
        lines.append(f"  Age {age:3d}: {count:>15,} ({format_percentage(count, total)})")
    lines.append("")
    
    # Age pair frequencies (top 30)
    lines.append("(AGE_0, AGE_1023) PAIR FREQUENCIES (Top 30)")
    lines.append("-" * 40)
    for (age_0, age_1023), count in stats['age_pair_0_1023'].most_common(30):
        lines.append(f"  ({age_0:3d}, {age_1023:3d}): {count:>12,} ({format_percentage(count, total)})")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistics for H5 sequence datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--h5_file", required=True,
                        help="Path to HDF5 file with input_ids")
    parser.add_argument("--output", default=None,
                        help="Output file for statistics report (default: stdout + h5_file.stats.txt)")
    parser.add_argument("--n_workers", type=int, default=16,
                        help="Number of parallel workers (default: 16)")
    parser.add_argument("--chunk_size", type=int, default=500000,
                        help="Chunk size for parallel processing (default: 500000)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.h5_file):
        logger.error(f"H5 file not found: {args.h5_file}")
        sys.exit(1)
    
    # Get dataset size
    with h5py.File(args.h5_file, 'r') as f:
        n_sequences = f['input_ids'].shape[0]
        shape = f['input_ids'].shape
        logger.info(f"Dataset shape: {shape}")
        logger.info(f"Total sequences: {n_sequences:,}")
    
    # Create chunks
    chunks = []
    for start in range(0, n_sequences, args.chunk_size):
        end = min(start + args.chunk_size, n_sequences)
        chunks.append((start, end))
    
    logger.info(f"Processing {len(chunks)} chunks of ~{args.chunk_size:,} sequences each")
    logger.info(f"Using {args.n_workers} parallel workers with vectorized processing")
    
    start_time = time.time()
    
    # Process chunks in parallel with tqdm progress bar
    all_stats = []
    
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {
            executor.submit(process_chunk_vectorized, args.h5_file, start, end): (start, end)
            for start, end in chunks
        }
        
        with tqdm(total=len(chunks), desc="Processing chunks", unit="chunk") as pbar:
            for future in as_completed(futures):
                start, end = futures[future]
                try:
                    stats = future.result()
                    all_stats.append(stats)
                    pbar.update(1)
                    pbar.set_postfix({
                        'sequences': f"{sum(s['total'] for s in all_stats):,}",
                        'matching': f"{sum(s['all_criteria'] for s in all_stats):,}"
                    })
                except Exception as e:
                    logger.error(f"Error processing chunk {start}-{end}: {e}")
                    pbar.update(1)
    
    # Merge all statistics
    logger.info("Merging statistics...")
    merged_stats = merge_stats(all_stats)
    
    elapsed = time.time() - start_time
    logger.info(f"Processing complete in {elapsed:.1f}s ({n_sequences/elapsed:,.0f} sequences/sec)")
    
    # Format report
    report = format_report(merged_stats)
    
    # Print to stdout
    print("\n" + report)
    
    # Save to file
    if args.output:
        output_path = args.output
    else:
        output_path = args.h5_file.replace('.h5', '_stats.txt')
        if output_path == args.h5_file:
            output_path = args.h5_file + '.stats.txt'
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to: {output_path}")
    
    # Also save detailed counters as CSV for further analysis
    csv_output = output_path.replace('.txt', '_age_pairs.csv')
    with open(csv_output, 'w') as f:
        f.write("age_0,age_1023,count\n")
        for (age_0, age_1023), count in sorted(merged_stats['age_pair_0_1023'].items()):
            f.write(f"{age_0},{age_1023},{count}\n")
    logger.info(f"Age pairs saved to: {csv_output}")
    
    # Save decade distributions as CSV
    decade_csv = output_path.replace('.txt', '_decades.csv')
    with open(decade_csv, 'w') as f:
        f.write("position,decade,count,percentage\n")
        total = merged_stats['total']
        for pos, key in [('0', 'decade_at_0'), ('6', 'decade_at_6'), ('1023', 'decade_at_1023')]:
            for decade in DECADE_LABELS:
                count = merged_stats[key].get(decade, 0)
                pct = 100 * count / total if total > 0 else 0
                f.write(f"{pos},{decade},{count},{pct:.4f}\n")
    logger.info(f"Decade distributions saved to: {decade_csv}")


if __name__ == "__main__":
    main()
