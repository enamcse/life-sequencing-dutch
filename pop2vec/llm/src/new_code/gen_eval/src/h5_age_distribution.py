#!/usr/bin/env python3
"""
H5 Age Distribution Analysis - Compute age statistics at each X00 token position.

Computes the age distribution at positions 0, 100, 200, ..., 900 in the sequences:
- Mean age
- Standard deviation
- Histogram/frequency counts

Usage:
    python h5_age_distribution.py --h5_file encoded.h5 --output age_stats.csv
    python h5_age_distribution.py --h5_file encoded.h5 --output age_stats.csv --n_workers 16
    
    # Custom positions
    python h5_age_distribution.py --h5_file encoded.h5 --positions 0,100,200,300,400,500,600,700,800,900
    
    # Generate plots
    python h5_age_distribution.py --h5_file encoded.h5 --output age_stats.csv --plot

The HDF5 file should have 'input_ids' with shape (N, 4, 1024):
    - input_ids[:, 0, :] = token IDs
    - input_ids[:, 1, :] = day count from genesis date
    - input_ids[:, 2, :] = person's age in years from birth
    - input_ids[:, 3, :] = related to previous token or not

Output:
    - age_stats.csv: Statistics for each position (mean, std, min, max, quartiles)
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

# Default positions to analyze (X00 for X in 0-9)
DEFAULT_POSITIONS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]


def process_chunk(
    h5_path: str,
    start_idx: int,
    end_idx: int,
    positions: List[int]
) -> Dict[int, List[int]]:
    """
    Process a chunk of sequences and collect ages at specified positions.
    
    Args:
        h5_path: Path to HDF5 file
        start_idx: Start index (inclusive)
        end_idx: End index (exclusive)
        positions: List of token positions to analyze
    
    Returns:
        Dict mapping position -> list of ages at that position
    """
    try:
        result = {pos: [] for pos in positions}
        
        with h5py.File(h5_path, 'r') as f:
            input_ids = f['input_ids']
            
            # Read the age channel (index 2) for this chunk
            # Shape: (chunk_size, seq_len)
            ages = input_ids[start_idx:end_idx, 2, :]
            
            for pos in positions:
                if pos < ages.shape[1]:
                    # Collect all ages at this position
                    ages_at_pos = ages[:, pos].tolist()
                    result[pos].extend(ages_at_pos)
        
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


def compute_statistics(ages_by_position: Dict[int, np.ndarray]) -> pd.DataFrame:
    """
    Compute summary statistics for each position.
    
    Returns DataFrame with columns:
        position, count, mean, std, min, q25, median, q75, max
    """
    rows = []
    
    for pos in sorted(ages_by_position.keys()):
        ages = ages_by_position[pos]
        
        # Filter out zero ages (might be padding)
        valid_ages = ages[ages > 0]
        
        if len(valid_ages) == 0:
            rows.append({
                'position': pos,
                'count': 0,
                'count_with_zeros': len(ages),
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'q25': np.nan,
                'median': np.nan,
                'q75': np.nan,
                'max': np.nan,
            })
        else:
            rows.append({
                'position': pos,
                'count': len(valid_ages),
                'count_with_zeros': len(ages),
                'mean': np.mean(valid_ages),
                'std': np.std(valid_ages),
                'min': np.min(valid_ages),
                'q25': np.percentile(valid_ages, 25),
                'median': np.median(valid_ages),
                'q75': np.percentile(valid_ages, 75),
                'max': np.max(valid_ages),
            })
    
    return pd.DataFrame(rows)


def compute_histograms(
    ages_by_position: Dict[int, np.ndarray],
    bin_width: int = 1
) -> pd.DataFrame:
    """
    Compute histogram data for each position.
    
    Returns DataFrame with columns:
        position, age, count, frequency
    """
    rows = []
    
    for pos in sorted(ages_by_position.keys()):
        ages = ages_by_position[pos]
        
        # Filter out zero ages
        valid_ages = ages[ages > 0]
        
        if len(valid_ages) == 0:
            continue
        
        # Count frequencies
        counter = Counter(valid_ages)
        total = len(valid_ages)
        
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
    """Generate histogram plots for each position."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Individual histograms
    for pos in positions:
        ages = ages_by_position.get(pos, np.array([]))
        valid_ages = ages[ages > 0]
        
        if len(valid_ages) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Histogram
        bins = np.arange(0, max(valid_ages) + 2, 1)
        ax.hist(valid_ages, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        
        # Add statistics
        mean_age = np.mean(valid_ages)
        std_age = np.std(valid_ages)
        ax.axvline(mean_age, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_age:.1f}')
        ax.axvline(mean_age - std_age, color='orange', linestyle=':', linewidth=1.5,
                   label=f'±1 Std: {std_age:.1f}')
        ax.axvline(mean_age + std_age, color='orange', linestyle=':', linewidth=1.5)
        
        ax.set_xlabel('Age (years)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Age Distribution at Token Position {pos}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add text with statistics
        stats_text = f'N={len(valid_ages):,}\nMean={mean_age:.1f}\nStd={std_age:.1f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'age_histogram_pos{pos}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved histogram: {output_path}")
    
    # Combined plot: mean and std across positions
    fig, ax = plt.subplots(figsize=(14, 6))
    
    means = []
    stds = []
    valid_positions = []
    
    for pos in positions:
        ages = ages_by_position.get(pos, np.array([]))
        valid_ages = ages[ages > 0]
        
        if len(valid_ages) > 0:
            valid_positions.append(pos)
            means.append(np.mean(valid_ages))
            stds.append(np.std(valid_ages))
    
    if valid_positions:
        x = np.arange(len(valid_positions))
        
        ax.errorbar(x, means, yerr=stds, fmt='o-', capsize=5, capthick=2,
                    markersize=10, linewidth=2, color='steelblue',
                    label='Mean ± Std')
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in valid_positions])
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Age (years)', fontsize=12)
        ax.set_title('Age Distribution Across Token Positions (Mean ± Std)', fontsize=14)
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
        valid_ages = ages[ages > 0]
        
        if len(valid_ages) > 0:
            hist, _ = np.histogram(valid_ages, bins=age_buckets)
            matrix[i, :] = hist / len(valid_ages)  # Normalize to frequency
    
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
        """
    )
    
    parser.add_argument("--h5_file", required=True, help="Path to HDF5 file")
    parser.add_argument("--output", default="age_stats.csv", 
                        help="Output path for statistics CSV (default: age_stats.csv)")
    parser.add_argument("--positions", type=str, default=None,
                        help="Comma-separated list of positions to analyze (default: 0,100,200,...,900)")
    parser.add_argument("--n_workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="Chunk size for processing (default: 100000)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate histogram plots")
    parser.add_argument("--plot_dir", default=None,
                        help="Output directory for plots (default: age_distribution_plots)")
    
    args = parser.parse_args()
    
    # Parse positions
    if args.positions:
        positions = [int(p.strip()) for p in args.positions.split(',')]
    else:
        positions = DEFAULT_POSITIONS
    
    logger.info(f"Analyzing positions: {positions}")
    
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
    
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {
            executor.submit(process_chunk, args.h5_file, start, end, positions): (start, end)
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
    stats_df = compute_statistics(ages_by_position)
    
    # Save statistics
    output_path = args.output
    stats_df.to_csv(output_path, index=False)
    logger.info(f"Saved statistics: {output_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("AGE DISTRIBUTION SUMMARY")
    logger.info("="*60)
    for _, row in stats_df.iterrows():
        logger.info(f"Position {int(row['position']):4d}: "
                   f"Mean={row['mean']:.1f}, Std={row['std']:.1f}, "
                   f"Range=[{row['min']:.0f}, {row['max']:.0f}], "
                   f"N={int(row['count']):,}")
    logger.info("="*60)
    
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
