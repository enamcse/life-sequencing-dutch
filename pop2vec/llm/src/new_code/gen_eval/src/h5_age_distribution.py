#!/usr/bin/env python3
"""
H5 Age Distribution Analysis - Compute age statistics at each X00 token position.

Computes the age distribution at positions 0, 100, 200, ..., 1000 in the sequences:
- Mean age
- Standard deviation  
- Percentiles (10th, 20th, ..., 90th) with counts for each percentile bucket

Note: For data export compliance, we report:
- Percentiles instead of min/max (10th to 90th)
- Count of people in each percentile bucket (must be >= 10 for export)

PAD Token Handling:
- If age=0, we check if token_id is also PAD (typically 0)
- Only exclude if both age=0 AND token_id=PAD_ID

Usage:
    python h5_age_distribution.py --h5_file encoded.h5 --output age_stats.csv
    python h5_age_distribution.py --h5_file encoded.h5 --output age_stats.csv --n_workers 16
    
    # Custom positions
    python h5_age_distribution.py --h5_file encoded.h5 --positions 0,100,200,300,400,500,600,700,800,900,1000
    
    # Generate plots
    python h5_age_distribution.py --h5_file encoded.h5 --output age_stats.csv --plot
    
    # Custom PAD token ID
    python h5_age_distribution.py --h5_file encoded.h5 --output age_stats.csv --pad_id 0

The HDF5 file should have 'input_ids' with shape (N, 4, 1024):
    - input_ids[:, 0, :] = token IDs
    - input_ids[:, 1, :] = day count from genesis date
    - input_ids[:, 2, :] = person's age in years from birth
    - input_ids[:, 3, :] = related to previous token or not

Output:
    - age_stats.csv: Statistics for each position (mean, std, percentiles with counts)
    - age_stats_percentile_buckets.csv: Count of people in each percentile bucket
    - age_histograms.csv: Full histogram data for each position
    - age_distribution_plots/: Directory with histogram plots (if --plot)
"""

import argparse
import logging
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Default positions to analyze (X00 for X in 0-10)
DEFAULT_POSITIONS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Percentiles to report (10th through 90th)
PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# Minimum count for export (data protection)
MIN_COUNT_FOR_EXPORT = 10


def process_chunk(
    h5_path: str,
    start_idx: int,
    end_idx: int,
    positions: List[int],
    pad_id: int = 0
) -> Dict[int, List[int]]:
    """
    Process a chunk of sequences and collect ages at specified positions.
    
    Handles PAD token detection: if age=0, check if token_id is also PAD.
    Only exclude if both age=0 AND token_id=pad_id.
    
    Args:
        h5_path: Path to HDF5 file
        start_idx: Start index (inclusive)
        end_idx: End index (exclusive)
        positions: List of token positions to analyze
        pad_id: PAD token ID (default: 0)
    
    Returns:
        Dict mapping position -> list of ages at that position (excluding PAD tokens)
    """
    try:
        result = {pos: [] for pos in positions}
        
        with h5py.File(h5_path, 'r') as f:
            input_ids = f['input_ids']
            
            # Read both token IDs (index 0) and ages (index 2) for this chunk
            # Shape: (chunk_size, seq_len)
            tokens = input_ids[start_idx:end_idx, 0, :]
            ages = input_ids[start_idx:end_idx, 2, :]
            
            for pos in positions:
                if pos < ages.shape[1]:
                    # Get ages and tokens at this position
                    ages_at_pos = ages[:, pos]
                    tokens_at_pos = tokens[:, pos]
                    
                    # Filter: exclude only if BOTH age=0 AND token=PAD
                    # This preserves actual age=0 tokens (e.g., birth events)
                    valid_mask = ~((ages_at_pos == 0) & (tokens_at_pos == pad_id))
                    valid_ages = ages_at_pos[valid_mask].tolist()
                    
                    result[pos].extend(valid_ages)
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing chunk [{start_idx}:{end_idx}]: {e}")
        return {pos: [] for pos in positions}


def merge_results(
    results_list: List[Dict[int, List[int]]],
    positions: List[int]
) -> Dict[int, np.ndarray]:
    """Merge results from multiple chunks."""
    merged = {pos: [] for pos in positions}
    
    for result in results_list:
        for pos in positions:
            merged[pos].extend(result.get(pos, []))
    
    # Convert to numpy arrays
    return {pos: np.array(ages) for pos, ages in merged.items()}


def compute_statistics(ages_by_position: Dict[int, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute summary statistics for each position.
    
    For data export compliance:
    - Uses percentiles (10th-90th) instead of min/max
    - Reports count of people in each percentile bucket
    - Flags if any bucket has < MIN_COUNT_FOR_EXPORT people
    
    Returns:
        Tuple of (stats_df, bucket_df):
        - stats_df: Statistics for each position (mean, std, percentiles)
        - bucket_df: Count of people in each percentile bucket
    """
    stats_rows = []
    bucket_rows = []
    
    for pos in sorted(ages_by_position.keys()):
        ages = ages_by_position[pos]
        
        if len(ages) == 0:
            row = {
                'position': pos,
                'total_count': 0,
                'valid_count': 0,
                'mean': np.nan,
                'std': np.nan,
            }
            for p in PERCENTILES:
                row[f'p{p}'] = np.nan
                row[f'p{p}_count'] = 0
            row['exportable'] = False
            stats_rows.append(row)
            continue
        
        # Compute basic statistics
        row = {
            'position': pos,
            'total_count': len(ages),
            'valid_count': len(ages),
            'mean': np.mean(ages),
            'std': np.std(ages),
        }
        
        # Compute percentiles
        percentile_values = {}
        for p in PERCENTILES:
            percentile_values[p] = np.percentile(ages, p)
            row[f'p{p}'] = percentile_values[p]
        
        # Compute counts in each percentile bucket
        # Buckets: [0, p10), [p10, p20), ..., [p80, p90), [p90, 100]
        bucket_counts = []
        sorted_ages = np.sort(ages)
        n = len(sorted_ages)
        
        # First bucket: below p10
        p10_idx = int(n * 0.10)
        bucket_counts.append(p10_idx)
        
        # Middle buckets: p10-p20, p20-p30, ..., p80-p90
        for i, p in enumerate(PERCENTILES[:-1]):
            lower_idx = int(n * p / 100)
            upper_idx = int(n * PERCENTILES[i + 1] / 100)
            bucket_counts.append(upper_idx - lower_idx)
        
        # Last bucket: above p90
        p90_idx = int(n * 0.90)
        bucket_counts.append(n - p90_idx)
        
        # Add counts to row
        bucket_labels = ['below_p10'] + [f'p{PERCENTILES[i]}_to_p{PERCENTILES[i+1]}' for i in range(len(PERCENTILES)-1)] + ['above_p90']
        for i, label in enumerate(bucket_labels):
            row[f'{label}_count'] = bucket_counts[i]
        
        # Check if all buckets have >= MIN_COUNT_FOR_EXPORT
        row['exportable'] = all(c >= MIN_COUNT_FOR_EXPORT for c in bucket_counts)
        row['min_bucket_count'] = min(bucket_counts)
        
        stats_rows.append(row)
        
        # Add to bucket_rows for detailed output
        for i, label in enumerate(bucket_labels):
            bucket_rows.append({
                'position': pos,
                'bucket': label,
                'count': bucket_counts[i],
                'meets_export_threshold': bucket_counts[i] >= MIN_COUNT_FOR_EXPORT,
            })
    
    return pd.DataFrame(stats_rows), pd.DataFrame(bucket_rows)


def compute_histograms(
    ages_by_position: Dict[int, np.ndarray],
    bin_width: int = 1
) -> pd.DataFrame:
    """
    Compute histogram data for each position.
    
    Note: PAD tokens are already filtered out in process_chunk,
    so we don't need to filter here. Age=0 entries are real tokens.
    
    Returns DataFrame with columns:
        position, age, count, frequency
    """
    rows = []
    
    for pos in sorted(ages_by_position.keys()):
        ages = ages_by_position[pos]
        
        if len(ages) == 0:
            continue
        
        # Count frequencies (age=0 is kept if it's a real token)
        counter = Counter(ages)
        total = len(ages)
        
        for age in sorted(counter.keys()):
            rows.append({
                'position': pos,
                'age': int(age),
                'count': counter[age],
                'frequency': counter[age] / total,
            })
    
    return pd.DataFrame(rows)


def plot_histograms(
    ages_by_position: Dict[int, np.ndarray],
    output_dir: str,
    positions: List[int]
):
    """Generate histogram plots for each position.
    
    Note: PAD tokens are already filtered in process_chunk,
    so all ages (including 0) are valid real tokens.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Individual histograms
    for pos in tqdm(positions, desc="Plotting histograms"):
        ages = ages_by_position.get(pos, np.array([]))
        
        if len(ages) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Histogram
        bins = np.arange(int(ages.min()), int(ages.max()) + 2, 1)
        ax.hist(ages, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        
        # Add statistics
        mean_age = np.mean(ages)
        std_age = np.std(ages)
        p10 = np.percentile(ages, 10)
        p90 = np.percentile(ages, 90)
        
        ax.axvline(mean_age, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_age:.1f}')
        ax.axvline(p10, color='green', linestyle=':', linewidth=1.5,
                   label=f'P10: {p10:.0f}')
        ax.axvline(p90, color='green', linestyle=':', linewidth=1.5,
                   label=f'P90: {p90:.0f}')
        
        ax.set_xlabel('Age (years)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Age Distribution at Token Position {pos}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add text with statistics
        stats_text = f'N={len(ages):,}\nMean={mean_age:.1f}\nStd={std_age:.1f}\nP10={p10:.0f}, P90={p90:.0f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'age_histogram_pos{pos}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Saved {len(positions)} individual histograms")
    
    # Combined plot: mean and std across positions
    fig, ax = plt.subplots(figsize=(14, 6))
    
    means = []
    stds = []
    p10s = []
    p90s = []
    valid_positions = []
    
    for pos in positions:
        ages = ages_by_position.get(pos, np.array([]))
        
        if len(ages) > 0:
            valid_positions.append(pos)
            means.append(np.mean(ages))
            stds.append(np.std(ages))
            p10s.append(np.percentile(ages, 10))
            p90s.append(np.percentile(ages, 90))
    
    if valid_positions:
        x = np.arange(len(valid_positions))
        
        ax.errorbar(x, means, yerr=stds, fmt='o-', capsize=5, capthick=2,
                    markersize=10, linewidth=2, color='steelblue',
                    label='Mean ± Std')
        
        # Also plot P10 and P90 as shaded region
        ax.fill_between(x, p10s, p90s, alpha=0.2, color='green', label='P10-P90 range')
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in valid_positions])
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Age (years)', fontsize=12)
        ax.set_title('Age Distribution Across Token Positions (Mean ± Std, P10-P90)', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'age_progression_summary.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved summary plot: {output_path}")
    
    # Heatmap of age distribution
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create matrix: positions x age buckets
    max_age = 110
    age_buckets = list(range(0, max_age + 1, 5))  # 5-year buckets
    
    matrix = np.zeros((len(positions), len(age_buckets) - 1))
    
    for i, pos in enumerate(positions):
        ages = ages_by_position.get(pos, np.array([]))
        
        if len(ages) > 0:
            hist, _ = np.histogram(ages, bins=age_buckets)
            matrix[i, :] = hist / len(ages)  # Normalize to frequency
    
    im = ax.imshow(matrix.T, aspect='auto', cmap='viridis', origin='lower')
    
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([str(p) for p in positions])
    ax.set_yticks(range(len(age_buckets) - 1))
    ax.set_yticklabels([f'{age_buckets[i]}-{age_buckets[i+1]-1}' for i in range(len(age_buckets) - 1)])
    
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Age Bucket (years)', fontsize=12)
    ax.set_title('Age Distribution Heatmap Across Token Positions', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Frequency', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'age_distribution_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved heatmap: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute age distribution at each X00 token position in H5 file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python h5_age_distribution.py --h5_file encoded.h5 --output age_stats.csv
    
    # With parallel processing
    python h5_age_distribution.py --h5_file encoded.h5 --output age_stats.csv --n_workers 16
    
    # Custom positions
    python h5_age_distribution.py --h5_file encoded.h5 --positions 0,100,200,500,1000
    
    # Generate plots
    python h5_age_distribution.py --h5_file encoded.h5 --output age_stats.csv --plot
    
    # Custom PAD token ID
    python h5_age_distribution.py --h5_file encoded.h5 --output age_stats.csv --pad_id 0
        """
    )
    
    parser.add_argument("--h5_file", required=True, help="Path to HDF5 file")
    parser.add_argument("--output", default="age_stats.csv", 
                        help="Output path for statistics CSV (default: age_stats.csv)")
    parser.add_argument("--positions", type=str, default=None,
                        help="Comma-separated list of positions to analyze (default: 0,100,200,...,1000)")
    parser.add_argument("--n_workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="Chunk size for processing (default: 100000)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate histogram plots")
    parser.add_argument("--plot_dir", default=None,
                        help="Output directory for plots (default: age_distribution_plots)")
    parser.add_argument("--pad_id", type=int, default=0,
                        help="PAD token ID (default: 0). Age=0 is only excluded if token_id is also PAD.")
    
    args = parser.parse_args()
    
    # Parse positions
    if args.positions:
        positions = [int(p.strip()) for p in args.positions.split(',')]
    else:
        positions = DEFAULT_POSITIONS
    
    logger.info(f"Analyzing positions: {positions}")
    logger.info(f"PAD token ID: {args.pad_id}")
    
    # Get dataset size
    with h5py.File(args.h5_file, 'r') as f:
        n_sequences = f['input_ids'].shape[0]
        seq_len = f['input_ids'].shape[2]
        logger.info(f"Dataset: {n_sequences:,} sequences, length {seq_len}")
    
    # Validate positions
    positions = [p for p in positions if p < seq_len]
    logger.info(f"Valid positions (< {seq_len}): {positions}")
    
    # Create chunks
    chunks = []
    for start in range(0, n_sequences, args.chunk_size):
        end = min(start + args.chunk_size, n_sequences)
        chunks.append((start, end))
    
    logger.info(f"Processing {len(chunks)} chunks with {args.n_workers} workers")
    
    # Process chunks in parallel
    start_time = time.time()
    all_results = []
    
    from functools import partial
    process_fn = partial(process_chunk, positions=positions, pad_id=args.pad_id)
    
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {
            executor.submit(process_chunk, args.h5_file, start, end, positions, args.pad_id): (start, end)
            for start, end in chunks
        }
        
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                pbar.update(1)
    
    elapsed = time.time() - start_time
    logger.info(f"Processing complete in {elapsed:.1f}s ({n_sequences/elapsed:,.0f} seq/s)")
    
    # Merge results
    ages_by_position = merge_results(all_results, positions)
    
    # Compute statistics
    logger.info("Computing statistics...")
    stats_df, bucket_df = compute_statistics(ages_by_position)
    
    # Save statistics
    output_path = args.output
    stats_df.to_csv(output_path, index=False)
    logger.info(f"Saved statistics: {output_path}")
    
    # Save bucket counts
    bucket_path = output_path.replace('.csv', '_percentile_buckets.csv')
    bucket_df.to_csv(bucket_path, index=False)
    logger.info(f"Saved percentile buckets: {bucket_path}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("AGE DISTRIBUTION SUMMARY (Export-Safe: Percentiles with Counts)")
    logger.info("="*70)
    for _, row in stats_df.iterrows():
        exportable_str = "✓ EXPORTABLE" if row.get('exportable', False) else f"✗ min_bucket={row.get('min_bucket_count', 0)}"
        logger.info(f"Position {int(row['position']):4d}: "
                   f"Mean={row['mean']:.1f}, Std={row['std']:.1f}, "
                   f"P10={row['p10']:.0f}, P50={row['p50']:.0f}, P90={row['p90']:.0f}, "
                   f"N={int(row['valid_count']):,} [{exportable_str}]")
    logger.info("="*70)
    
    # Compute and save histograms
    logger.info("Computing histograms...")
    hist_df = compute_histograms(ages_by_position)
    
    hist_path = output_path.replace('.csv', '_histograms.csv')
    hist_df.to_csv(hist_path, index=False)
    logger.info(f"Saved histograms: {hist_path}")
    
    # Generate plots if requested
    if args.plot:
        plot_dir = args.plot_dir or os.path.join(os.path.dirname(output_path) or '.', 'age_distribution_plots')
        logger.info(f"Generating plots in: {plot_dir}")
        plot_histograms(ages_by_position, plot_dir, positions)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
