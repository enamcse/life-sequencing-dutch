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
    
    # Use custom end position instead of 1023
    python h5_sequence_statistics.py --h5_file encoded.h5 --end_pos 800
    
    # Scan a range of positions (e.g., 1000-1023) to find where sequences really end
    python h5_sequence_statistics.py --h5_file encoded.h5 --scan_range 1000-1023
    
    # Find the real last non-zero position for each sequence
    python h5_sequence_statistics.py --h5_file encoded.h5 --find_real_end

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


def process_chunk_vectorized(h5_path: str, start_idx: int, end_idx: int, 
                              end_pos: int = 1023, scan_positions: List[int] = None,
                              find_real_end: bool = False) -> Dict:
    """
    Process a chunk of sequences using vectorized numpy operations.
    
    This is MUCH faster than the loop-based version for large datasets.
    Memory-efficient: only reads required columns (indices 0, 6, end_pos).
    
    Args:
        h5_path: Path to HDF5 file
        start_idx: Start index (inclusive)
        end_idx: End index (exclusive)
        end_pos: Position to use as "end" (default: 1023)
        scan_positions: List of positions to scan for decade stats (e.g., [1000, 1001, ..., 1023])
        find_real_end: If True, find the actual last non-zero position
    
    Returns:
        Dictionary with statistics for this chunk
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            # Memory optimization: Only read the specific columns we need
            input_ids = f['input_ids']
            n_sequences = end_idx - start_idx
            
            # Read only token stream (channel 0) at end_pos
            token_end = input_ids[start_idx:end_idx, 0, end_pos].astype(np.int32)
            
            # Read only age stream (channel 2) at positions 0, 6, end_pos
            age_0 = input_ids[start_idx:end_idx, 2, 0].astype(np.int32)
            age_6 = input_ids[start_idx:end_idx, 2, 6].astype(np.int32)
            age_end = input_ids[start_idx:end_idx, 2, end_pos].astype(np.int32)
            
            # Vectorized criteria checks
            childhood_start = (age_6 >= 0) & (age_6 <= 9)
            full_length = (token_end != 0) | (age_end != 0)
            last_3_decades = (age_end >= 70) & (age_end <= 99)
            
            # Count criteria
            stats = {
                'total': n_sequences,
                'end_pos': end_pos,
                'childhood_start': int(childhood_start.sum()),
                'full_length': int(full_length.sum()),
                'last_3_decades': int(last_3_decades.sum()),
                'all_criteria': int((childhood_start & full_length & last_3_decades).sum()),
                'childhood_and_full': int((childhood_start & full_length).sum()),
                'childhood_and_last3': int((childhood_start & last_3_decades).sum()),
                'full_and_last3': int((full_length & last_3_decades).sum()),
                'token_end_zero': int((token_end == 0).sum()),
                'token_end_nonzero': int((token_end != 0).sum()),
            }
            
            # Age distributions using np.unique for efficiency
            age_0_unique, age_0_counts = np.unique(age_0, return_counts=True)
            age_6_unique, age_6_counts = np.unique(age_6, return_counts=True)
            age_end_unique, age_end_counts = np.unique(age_end, return_counts=True)
            
            stats['age_at_0'] = Counter(dict(zip(age_0_unique.tolist(), age_0_counts.tolist())))
            stats['age_at_6'] = Counter(dict(zip(age_6_unique.tolist(), age_6_counts.tolist())))
            stats['age_at_end'] = Counter(dict(zip(age_end_unique.tolist(), age_end_counts.tolist())))
            
            # Decade distributions using vectorized binning
            decade_idx_0 = get_decade_vectorized(age_0)
            decade_idx_6 = get_decade_vectorized(age_6)
            decade_idx_end = get_decade_vectorized(age_end)
            
            stats['decade_at_0'] = Counter()
            stats['decade_at_6'] = Counter()
            stats['decade_at_end'] = Counter()
            
            for idx in range(len(DECADE_LABELS)):
                label = DECADE_LABELS[idx]
                stats['decade_at_0'][label] = int((decade_idx_0 == idx).sum())
                stats['decade_at_6'][label] = int((decade_idx_6 == idx).sum())
                stats['decade_at_end'][label] = int((decade_idx_end == idx).sum())
            
            # Age pair frequencies
            pair_keys = age_0.astype(np.int64) * 1000 + age_end.astype(np.int64)
            pair_unique, pair_counts = np.unique(pair_keys, return_counts=True)
            
            stats['age_pair_0_end'] = Counter()
            for key, count in zip(pair_unique.tolist(), pair_counts.tolist()):
                a0 = key // 1000
                a_end = key % 1000
                stats['age_pair_0_end'][(a0, a_end)] = count
            
            # Scan multiple positions if requested
            if scan_positions is not None:
                stats['position_stats'] = {}
                for pos in scan_positions:
                    if pos > input_ids.shape[2] - 1:
                        continue
                    
                    token_at_pos = input_ids[start_idx:end_idx, 0, pos].astype(np.int32)
                    age_at_pos = input_ids[start_idx:end_idx, 2, pos].astype(np.int32)
                    
                    decade_idx = get_decade_vectorized(age_at_pos)
                    
                    pos_stats = {
                        'token_zero': int((token_at_pos == 0).sum()),
                        'token_nonzero': int((token_at_pos != 0).sum()),
                        'age_zero': int((age_at_pos == 0).sum()),
                        'age_nonzero': int((age_at_pos != 0).sum()),
                        'decades': Counter(),
                    }
                    
                    for idx in range(len(DECADE_LABELS)):
                        label = DECADE_LABELS[idx]
                        pos_stats['decades'][label] = int((decade_idx == idx).sum())
                    
                    stats['position_stats'][pos] = pos_stats
            
            # Find real end position if requested
            if find_real_end:
                # Read token stream for positions from end_pos backwards to find last non-zero
                # This is more expensive but gives us the actual sequence lengths
                # Read a window: [end_pos-100, end_pos+1] to find where sequences actually end
                window_start = max(0, end_pos - 200)
                tokens_window = input_ids[start_idx:end_idx, 0, window_start:end_pos+1].astype(np.int32)
                
                # Find last non-zero position for each sequence
                # Create a mask of non-zero positions
                nonzero_mask = tokens_window != 0
                
                # Find last non-zero index in the window for each sequence
                # We reverse and find first True, then convert back
                last_nonzero_in_window = np.zeros(n_sequences, dtype=np.int32)
                for i in range(n_sequences):
                    nonzero_indices = np.where(nonzero_mask[i])[0]
                    if len(nonzero_indices) > 0:
                        last_nonzero_in_window[i] = window_start + nonzero_indices[-1]
                    else:
                        last_nonzero_in_window[i] = 0  # All zeros in window
                
                real_end_unique, real_end_counts = np.unique(last_nonzero_in_window, return_counts=True)
                stats['real_end_positions'] = Counter(dict(zip(real_end_unique.tolist(), real_end_counts.tolist())))
        
        return stats
        
    except Exception as e:
        # Return empty stats on error so we can continue
        logger.error(f"Error in chunk {start_idx}-{end_idx}: {e}")
        return {
            'total': 0, 'end_pos': end_pos, 'childhood_start': 0, 'full_length': 0,
            'last_3_decades': 0, 'all_criteria': 0, 'childhood_and_full': 0,
            'childhood_and_last3': 0, 'full_and_last3': 0,
            'token_end_zero': 0, 'token_end_nonzero': 0,
            'age_at_0': Counter(), 'age_at_6': Counter(), 'age_at_end': Counter(),
            'decade_at_0': Counter(), 'decade_at_6': Counter(), 'decade_at_end': Counter(),
            'age_pair_0_end': Counter(), 'position_stats': {}, 'real_end_positions': Counter(),
        }


def merge_stats(stats_list: List[Dict], end_pos: int = 1023) -> Dict:
    """Merge statistics from multiple chunks."""
    merged = {
        'total': 0,
        'end_pos': end_pos,
        'childhood_start': 0,
        'full_length': 0,
        'last_3_decades': 0,
        'all_criteria': 0,
        'age_at_0': Counter(),
        'age_at_6': Counter(),
        'age_at_end': Counter(),
        'decade_at_0': Counter(),
        'decade_at_6': Counter(),
        'decade_at_end': Counter(),
        'age_pair_0_end': Counter(),
        'token_end_zero': 0,
        'token_end_nonzero': 0,
        'childhood_and_full': 0,
        'childhood_and_last3': 0,
        'full_and_last3': 0,
        'position_stats': {},  # For scan_range
        'real_end_positions': Counter(),  # For find_real_end
    }
    
    for stats in stats_list:
        merged['total'] += stats['total']
        merged['childhood_start'] += stats['childhood_start']
        merged['full_length'] += stats['full_length']
        merged['last_3_decades'] += stats['last_3_decades']
        merged['all_criteria'] += stats['all_criteria']
        merged['token_end_zero'] += stats.get('token_end_zero', 0)
        merged['token_end_nonzero'] += stats.get('token_end_nonzero', 0)
        merged['childhood_and_full'] += stats['childhood_and_full']
        merged['childhood_and_last3'] += stats['childhood_and_last3']
        merged['full_and_last3'] += stats['full_and_last3']
        
        merged['age_at_0'].update(stats['age_at_0'])
        merged['age_at_6'].update(stats['age_at_6'])
        merged['age_at_end'].update(stats.get('age_at_end', Counter()))
        merged['decade_at_0'].update(stats['decade_at_0'])
        merged['decade_at_6'].update(stats['decade_at_6'])
        merged['decade_at_end'].update(stats.get('decade_at_end', Counter()))
        merged['age_pair_0_end'].update(stats.get('age_pair_0_end', Counter()))
        
        # Merge position_stats (from scan_range)
        for pos, pos_stats in stats.get('position_stats', {}).items():
            if pos not in merged['position_stats']:
                merged['position_stats'][pos] = {
                    'token_zero': 0,
                    'token_nonzero': 0,
                    'age_zero': 0,
                    'age_nonzero': 0,
                    'decades': Counter(),
                }
            merged['position_stats'][pos]['token_zero'] += pos_stats['token_zero']
            merged['position_stats'][pos]['token_nonzero'] += pos_stats['token_nonzero']
            merged['position_stats'][pos]['age_zero'] += pos_stats['age_zero']
            merged['position_stats'][pos]['age_nonzero'] += pos_stats['age_nonzero']
            merged['position_stats'][pos]['decades'].update(pos_stats['decades'])
        
        # Merge real_end_positions (from find_real_end)
        merged['real_end_positions'].update(stats.get('real_end_positions', Counter()))
    
    return merged


def format_percentage(count: int, total: int) -> str:
    """Format count as percentage."""
    if total == 0:
        return "0.00%"
    return f"{100 * count / total:.4f}%"


def format_report(stats: Dict) -> str:
    """Format statistics into a human-readable report."""
    total = stats['total']
    end_pos = stats.get('end_pos', 1023)
    
    lines = []
    lines.append("=" * 80)
    lines.append("H5 SEQUENCE STATISTICS REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total sequences:                    {total:,}")
    lines.append(f"End position used:                  {end_pos}")
    lines.append("")
    
    # Main criteria
    lines.append("MAIN CRITERIA")
    lines.append("-" * 40)
    lines.append(f"1. Childhood start (age at idx 6 in 0-9): {stats['childhood_start']:,} ({format_percentage(stats['childhood_start'], total)})")
    lines.append(f"2. Full length (non-zero at idx {end_pos}):    {stats['full_length']:,} ({format_percentage(stats['full_length'], total)})")
    lines.append(f"3. Last 3 decades (age {end_pos} in 70-99):    {stats['last_3_decades']:,} ({format_percentage(stats['last_3_decades'], total)})")
    lines.append("")
    
    # Criteria combinations
    lines.append("CRITERIA COMBINATIONS")
    lines.append("-" * 40)
    lines.append(f"Childhood + Full length:           {stats['childhood_and_full']:,} ({format_percentage(stats['childhood_and_full'], total)})")
    lines.append(f"Childhood + Last 3 decades:        {stats['childhood_and_last3']:,} ({format_percentage(stats['childhood_and_last3'], total)})")
    lines.append(f"Full length + Last 3 decades:      {stats['full_and_last3']:,} ({format_percentage(stats['full_and_last3'], total)})")
    lines.append(f"ALL THREE CRITERIA:                {stats['all_criteria']:,} ({format_percentage(stats['all_criteria'], total)})")
    lines.append("")
    
    # Token at end position
    lines.append(f"TOKEN AT INDEX {end_pos}")
    lines.append("-" * 40)
    lines.append(f"Zero (padding):     {stats['token_end_zero']:,} ({format_percentage(stats['token_end_zero'], total)})")
    lines.append(f"Non-zero (real):    {stats['token_end_nonzero']:,} ({format_percentage(stats['token_end_nonzero'], total)})")
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
    
    lines.append(f"DECADE DISTRIBUTION AT INDEX {end_pos} (End position)")
    lines.append("-" * 40)
    for decade in decade_order:
        count = stats['decade_at_end'].get(decade, 0)
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
    
    lines.append(f"AGE DISTRIBUTION AT INDEX {end_pos} (Top 20)")
    lines.append("-" * 40)
    for age, count in stats['age_at_end'].most_common(20):
        lines.append(f"  Age {age:3d}: {count:>15,} ({format_percentage(count, total)})")
    lines.append("")
    
    # Age pair frequencies (top 30)
    lines.append(f"(AGE_0, AGE_{end_pos}) PAIR FREQUENCIES (Top 30)")
    lines.append("-" * 40)
    for (age_0, age_end), count in stats['age_pair_0_end'].most_common(30):
        lines.append(f"  ({age_0:3d}, {age_end:3d}): {count:>12,} ({format_percentage(count, total)})")
    lines.append("")
    
    # Position scan results (if available)
    if stats.get('position_stats'):
        lines.append("=" * 80)
        lines.append("POSITION SCAN RESULTS")
        lines.append("=" * 80)
        lines.append("")
        
        for pos in sorted(stats['position_stats'].keys()):
            pos_stats = stats['position_stats'][pos]
            lines.append(f"POSITION {pos}")
            lines.append("-" * 40)
            lines.append(f"  Token zero:     {pos_stats['token_zero']:,} ({format_percentage(pos_stats['token_zero'], total)})")
            lines.append(f"  Token non-zero: {pos_stats['token_nonzero']:,} ({format_percentage(pos_stats['token_nonzero'], total)})")
            lines.append(f"  Age zero:       {pos_stats['age_zero']:,} ({format_percentage(pos_stats['age_zero'], total)})")
            lines.append(f"  Age non-zero:   {pos_stats['age_nonzero']:,} ({format_percentage(pos_stats['age_nonzero'], total)})")
            lines.append("  Decade distribution:")
            for decade in decade_order:
                count = pos_stats['decades'].get(decade, 0)
                if count > 0:
                    lines.append(f"    {decade:10s}: {count:>12,} ({format_percentage(count, total)})")
            lines.append("")
    
    # Real end position distribution (if available)
    if stats.get('real_end_positions'):
        lines.append("=" * 80)
        lines.append("REAL END POSITION DISTRIBUTION (last non-zero token position)")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Top 30 most common last non-zero positions:")
        lines.append("-" * 40)
        for pos, count in stats['real_end_positions'].most_common(30):
            lines.append(f"  Position {pos:4d}: {count:>12,} ({format_percentage(count, total)})")
        lines.append("")
        
        # Summary stats
        positions = list(stats['real_end_positions'].keys())
        if positions:
            min_pos = min(positions)
            max_pos = max(positions)
            # Weighted average
            total_weighted = sum(pos * count for pos, count in stats['real_end_positions'].items())
            avg_pos = total_weighted / total if total > 0 else 0
            lines.append(f"  Min position: {min_pos}")
            lines.append(f"  Max position: {max_pos}")
            lines.append(f"  Avg position: {avg_pos:.1f}")
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
    parser.add_argument("--n_workers", type=int, default=8,
                        help="Number of parallel workers (default: 8, use fewer for memory safety)")
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="Chunk size for parallel processing (default: 100000)")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential processing (slower but memory-safe)")
    parser.add_argument("--end_pos", type=int, default=1023,
                        help="End position to check (default: 1023)")
    parser.add_argument("--scan_range", type=str, default=None,
                        help="Scan a range of positions, e.g., '1000-1023' or '900-1023'")
    parser.add_argument("--find_real_end", action="store_true",
                        help="Find the actual last non-zero position for each sequence")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.h5_file):
        logger.error(f"H5 file not found: {args.h5_file}")
        sys.exit(1)
    
    # Parse scan_range if provided
    scan_positions = None
    if args.scan_range:
        try:
            start_pos, end_pos = map(int, args.scan_range.split('-'))
            scan_positions = list(range(start_pos, end_pos + 1))
            logger.info(f"Scanning positions {start_pos} to {end_pos} ({len(scan_positions)} positions)")
        except ValueError:
            logger.error(f"Invalid scan_range format: {args.scan_range}. Use 'start-end', e.g., '1000-1023'")
            sys.exit(1)
    
    # Get dataset size
    with h5py.File(args.h5_file, 'r') as f:
        n_sequences = f['input_ids'].shape[0]
        shape = f['input_ids'].shape
        logger.info(f"Dataset shape: {shape}")
        logger.info(f"Total sequences: {n_sequences:,}")
        logger.info(f"End position: {args.end_pos}")
        if args.find_real_end:
            logger.info("Finding real end positions (last non-zero token)")
    
    # Create chunks
    chunks = []
    for start in range(0, n_sequences, args.chunk_size):
        end = min(start + args.chunk_size, n_sequences)
        chunks.append((start, end))
    
    logger.info(f"Processing {len(chunks)} chunks of ~{args.chunk_size:,} sequences each")
    
    start_time = time.time()
    all_stats = []
    
    # Create partial function with the extra arguments
    process_func = partial(
        process_chunk_vectorized,
        end_pos=args.end_pos,
        scan_positions=scan_positions,
        find_real_end=args.find_real_end
    )
    
    if args.sequential:
        # Sequential processing - slower but guaranteed memory safe
        logger.info("Using sequential processing (memory-safe mode)")
        
        for start, end in tqdm(chunks, desc="Processing chunks", unit="chunk"):
            try:
                stats = process_func(args.h5_file, start, end)
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"Error processing chunk {start}-{end}: {e}")
    else:
        # Parallel processing with controlled memory
        logger.info(f"Using {args.n_workers} parallel workers")
        
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(process_func, args.h5_file, start, end): (start, end)
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
    merged_stats = merge_stats(all_stats, end_pos=args.end_pos)
    
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
    
    end_pos = args.end_pos
    
    # Also save detailed counters as CSV for further analysis
    csv_output = output_path.replace('.txt', '_age_pairs.csv')
    with open(csv_output, 'w') as f:
        f.write(f"age_0,age_{end_pos},count\n")
        for (age_0, age_end), count in sorted(merged_stats['age_pair_0_end'].items()):
            f.write(f"{age_0},{age_end},{count}\n")
    logger.info(f"Age pairs saved to: {csv_output}")
    
    # Save decade distributions as CSV
    decade_csv = output_path.replace('.txt', '_decades.csv')
    with open(decade_csv, 'w') as f:
        f.write("position,decade,count,percentage\n")
        total = merged_stats['total']
        for pos, key in [('0', 'decade_at_0'), ('6', 'decade_at_6'), (str(end_pos), 'decade_at_end')]:
            for decade in DECADE_LABELS:
                count = merged_stats[key].get(decade, 0)
                pct = 100 * count / total if total > 0 else 0
                f.write(f"{pos},{decade},{count},{pct:.4f}\n")
    logger.info(f"Decade distributions saved to: {decade_csv}")
    
    # Save position scan results if available
    if merged_stats.get('position_stats'):
        scan_csv = output_path.replace('.txt', '_position_scan.csv')
        with open(scan_csv, 'w') as f:
            f.write("position,token_zero,token_nonzero,age_zero,age_nonzero")
            for decade in DECADE_LABELS:
                f.write(f",{decade}")
            f.write("\n")
            
            for pos in sorted(merged_stats['position_stats'].keys()):
                pos_stats = merged_stats['position_stats'][pos]
                f.write(f"{pos},{pos_stats['token_zero']},{pos_stats['token_nonzero']}")
                f.write(f",{pos_stats['age_zero']},{pos_stats['age_nonzero']}")
                for decade in DECADE_LABELS:
                    f.write(f",{pos_stats['decades'].get(decade, 0)}")
                f.write("\n")
        logger.info(f"Position scan results saved to: {scan_csv}")
    
    # Save real end positions if available
    if merged_stats.get('real_end_positions'):
        real_end_csv = output_path.replace('.txt', '_real_end_positions.csv')
        with open(real_end_csv, 'w') as f:
            f.write("position,count,percentage\n")
            total = merged_stats['total']
            for pos, count in sorted(merged_stats['real_end_positions'].items()):
                pct = 100 * count / total if total > 0 else 0
                f.write(f"{pos},{count},{pct:.4f}\n")
        logger.info(f"Real end positions saved to: {real_end_csv}")


if __name__ == "__main__":
    main()
