#!/usr/bin/env python3
"""
Compute Statistics (CPU Phase)

Reads generated sequences from Parquet and computes all statistics.
Outputs two CSV files:
    1. statistics_full.csv - Full data with per-person columns
    2. statistics_summary.csv - Aggregated data only (no per-person columns)

Output CSV structure:
    - Each prefix_len forms a block
    - Block has 12 comparison rows (6 comparisons × with/without PAD)
    - Block has V token frequency rows
    - Full CSV columns: prefix_len, row_type, token_id, token, p0_num, p0_den, ..., total_num, total_den
    - Summary CSV columns: prefix_len, row_type, token_id, token, total_num, total_den, rate

PAD token exclusion modes:
    - 'seq1': Exclude position if PAD in first sequence (ground truth)
    - 'seq2': Exclude position if PAD in second sequence (generated)
    - 'both': Exclude position if PAD in either sequence (current default)

Comparison types:
    1-2. ordered_self_with_pad / ordered_self_no_pad
    3-4. unordered_self_with_pad / unordered_self_no_pad
    5-6. ordered_buddy_with_pad / ordered_buddy_no_pad
    7-8. unordered_buddy_with_pad / unordered_buddy_no_pad
    9-10. ordered_next_with_pad / ordered_next_no_pad
    11-12. unordered_next_with_pad / unordered_next_no_pad

Usage:
    python compute_statistics.py --config run_config.yaml
"""

import argparse
import json
import logging
import numpy as np
import os
import time
import yaml
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

import pandas as pd
import pyarrow.parquet as pq

# Logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StatsConfig:
    """Configuration for statistics computation."""
    model_name: str
    sequences_path: str
    statistics_path: str  # Full CSV with per-person columns
    statistics_summary_path: str  # Summary CSV without per-person columns
    vocab_path: str
    output_dir: str
    
    # PAD token ID (loaded from metadata or vocab)
    pad_id: int = 0
    
    # Include all tokens or just top N
    top_n_tokens: int = 0  # 0 = all tokens
    
    # PAD token exclusion mode: 'seq1', 'seq2', or 'both'
    pad_exclusion_mode: str = 'both'


# =============================================================================
# Statistics Functions
# =============================================================================

def parse_tokens(token_str: str) -> List[int]:
    """Parse comma-separated token string to list of ints."""
    if not token_str or pd.isna(token_str) or token_str == '':
        return []
    return [int(t) for t in str(token_str).split(',')]


def compute_ordered_match(
    seq1: List[int], 
    seq2: List[int], 
    exclude_pad: bool = False, 
    pad_id: int = 0,
    pad_exclusion_mode: str = 'both'
) -> Tuple[int, int]:
    """
    Compute ordered (position-wise) matches.
    
    Args:
        seq1: First sequence (ground truth)
        seq2: Second sequence (generated)
        exclude_pad: If True, exclude PAD tokens based on pad_exclusion_mode
        pad_id: PAD token ID
        pad_exclusion_mode: 'seq1' (exclude if PAD in seq1), 
                           'seq2' (exclude if PAD in seq2),
                           'both' (exclude if PAD in either)
    
    Returns:
        (num_matches, total_positions)
    """
    if exclude_pad:
        if pad_exclusion_mode == 'seq1':
            # Exclude positions where seq1 has PAD
            pairs = [(a, b) for a, b in zip(seq1, seq2) if a != pad_id]
        elif pad_exclusion_mode == 'seq2':
            # Exclude positions where seq2 has PAD
            pairs = [(a, b) for a, b in zip(seq1, seq2) if b != pad_id]
        else:  # 'both'
            # Exclude positions where either has PAD
            pairs = [(a, b) for a, b in zip(seq1, seq2) if a != pad_id and b != pad_id]
        
        if not pairs:
            return 0, 0
        matches = sum(1 for a, b in pairs if a == b)
        return matches, len(pairs)
    else:
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0, 0
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        return matches, min_len


def compute_unordered_match(
    seq1: List[int], 
    seq2: List[int], 
    exclude_pad: bool = False, 
    pad_id: int = 0,
    pad_exclusion_mode: str = 'both'
) -> Tuple[int, int]:
    """
    Compute unordered (multiset) matches.
    
    Args:
        seq1: First sequence (ground truth)
        seq2: Second sequence (generated)
        exclude_pad: If True, exclude PAD tokens based on pad_exclusion_mode
        pad_id: PAD token ID
        pad_exclusion_mode: 'seq1' (exclude PAD only from seq1), 
                           'seq2' (exclude PAD only from seq2),
                           'both' (exclude PAD from both)
    
    Returns:
        (num_matches, total_tokens_in_seq1)
    """
    if exclude_pad:
        if pad_exclusion_mode == 'seq1':
            seq1 = [t for t in seq1 if t != pad_id]
        elif pad_exclusion_mode == 'seq2':
            seq2 = [t for t in seq2 if t != pad_id]
        else:  # 'both'
            seq1 = [t for t in seq1 if t != pad_id]
            seq2 = [t for t in seq2 if t != pad_id]
    
    if len(seq1) == 0:
        return 0, 0
    
    counter1 = Counter(seq1)
    counter2 = Counter(seq2)
    
    # Intersection count (min of each token)
    intersection = sum((counter1 & counter2).values())
    return intersection, len(seq1)


def compute_token_frequencies(
    tokens_list: List[List[int]], 
    exclude_pad: bool = False, 
    pad_id: int = 0
) -> Tuple[Counter, int]:
    """
    Compute token frequencies across multiple sequences.
    
    Args:
        tokens_list: List of token sequences
        exclude_pad: If True, exclude PAD tokens
        pad_id: PAD token ID
    
    Returns:
        (Counter of token frequencies, total tokens)
    """
    counter = Counter()
    total = 0
    
    for tokens in tokens_list:
        if exclude_pad:
            tokens = [t for t in tokens if t != pad_id]
        counter.update(tokens)
        total += len(tokens)
    
    return counter, total


# =============================================================================
# Main Statistics Computation
# =============================================================================

class StatisticsComputer:
    """Computes all statistics from generated sequences."""
    
    # Row types in order
    ROW_TYPES = [
        'ordered_self_with_pad',
        'ordered_self_no_pad',
        'unordered_self_with_pad',
        'unordered_self_no_pad',
        'ordered_buddy_with_pad',
        'ordered_buddy_no_pad',
        'unordered_buddy_with_pad',
        'unordered_buddy_no_pad',
        'ordered_next_with_pad',
        'ordered_next_no_pad',
        'unordered_next_with_pad',
        'unordered_next_no_pad',
    ]
    
    def __init__(self, config: StatsConfig):
        self.config = config
        
        # Load vocabulary for token names
        logger.info(f"Loading vocabulary: {config.vocab_path}")
        self.vocab_df = pd.read_csv(config.vocab_path)
        self.id_to_token = dict(zip(self.vocab_df['ID'], self.vocab_df['TOKEN']))
        self.all_token_ids = sorted(self.vocab_df['ID'].tolist())
        self.vocab_size = len(self.all_token_ids)
        logger.info(f"Vocabulary size: {self.vocab_size}")
        logger.info(f"PAD exclusion mode: {config.pad_exclusion_mode}")
        
        # Load metadata if exists
        metadata_path = os.path.join(config.output_dir, 'generation_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.pad_id = metadata.get('pad_id', config.pad_id)
            logger.info(f"PAD token ID from metadata: {self.pad_id}")
        else:
            self.pad_id = config.pad_id
    
    def _compute_comparison_stats(
        self,
        df: pd.DataFrame,
        prefix_len: int,
        n_people: int
    ) -> List[Dict]:
        """Compute all 12 comparison statistics for a prefix block."""
        rows = []
        
        for row_type in self.ROW_TYPES:
            # Parse row type
            parts = row_type.split('_')
            is_ordered = parts[0] == 'ordered'
            comparison_target = parts[1]  # self, buddy, next
            include_pad = parts[-1] == 'pad' and parts[-2] == 'with'
            exclude_pad = not include_pad
            
            # Initialize accumulators for each person
            person_stats = {i: {'num': 0, 'den': 0} for i in range(n_people)}
            
            # Process each record
            for _, record in df.iterrows():
                person_idx = record['person_idx']
                
                generated = parse_tokens(record['generated_tokens'])
                
                # Select comparison target
                if comparison_target == 'self':
                    target = parse_tokens(record['original_tokens'])
                elif comparison_target == 'buddy':
                    target = parse_tokens(record['buddy_tokens'])
                else:  # next
                    target = parse_tokens(record['next_tokens'])
                
                # Compute match
                if is_ordered:
                    num, den = compute_ordered_match(
                        target, generated, exclude_pad, self.pad_id,
                        self.config.pad_exclusion_mode
                    )
                else:
                    num, den = compute_unordered_match(
                        target, generated, exclude_pad, self.pad_id,
                        self.config.pad_exclusion_mode
                    )
                
                person_stats[person_idx]['num'] += num
                person_stats[person_idx]['den'] += den
            
            # Build row
            row = {
                'prefix_len': prefix_len,
                'row_type': row_type,
                'token_id': None,
                'token': None,
            }
            
            total_num = 0
            total_den = 0
            
            for i in range(n_people):
                row[f'p{i}_num'] = person_stats[i]['num']
                row[f'p{i}_den'] = person_stats[i]['den']
                total_num += person_stats[i]['num']
                total_den += person_stats[i]['den']
            
            row['total_num'] = total_num
            row['total_den'] = total_den
            
            rows.append(row)
        
        return rows
    
    def _compute_token_frequency_stats(
        self,
        df: pd.DataFrame,
        prefix_len: int,
        n_people: int
    ) -> List[Dict]:
        """Compute token frequency statistics for a prefix block."""
        rows = []
        
        # Collect generated tokens per person
        person_tokens = {i: [] for i in range(n_people)}
        
        for _, record in df.iterrows():
            person_idx = record['person_idx']
            generated = parse_tokens(record['generated_tokens'])
            person_tokens[person_idx].append(generated)
        
        # Compute frequencies per person
        person_freqs = {}
        person_totals = {}
        
        for i in range(n_people):
            freq, total = compute_token_frequencies(person_tokens[i])
            person_freqs[i] = freq
            person_totals[i] = total
        
        # Determine which tokens to include
        if self.config.top_n_tokens > 0:
            # Aggregate all frequencies to find top tokens
            all_freqs = Counter()
            for i in range(n_people):
                all_freqs.update(person_freqs[i])
            
            top_tokens = [tid for tid, _ in all_freqs.most_common(self.config.top_n_tokens)]
        else:
            top_tokens = self.all_token_ids
        
        # Build rows for each token
        for token_id in top_tokens:
            row = {
                'prefix_len': prefix_len,
                'row_type': 'token_frequency',
                'token_id': token_id,
                'token': self.id_to_token.get(token_id, f'UNK_{token_id}'),
            }
            
            total_num = 0
            total_den = 0
            
            for i in range(n_people):
                num = person_freqs[i].get(token_id, 0)
                den = person_totals[i]
                row[f'p{i}_num'] = num
                row[f'p{i}_den'] = den
                total_num += num
                total_den += den
            
            row['total_num'] = total_num
            row['total_den'] = total_den
            
            rows.append(row)
        
        return rows
    
    def compute(self):
        """Compute all statistics and save to CSV."""
        logger.info("="*60)
        logger.info(f"Computing Statistics: {self.config.model_name}")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Load sequences
        logger.info(f"Loading sequences: {self.config.sequences_path}")
        df = pd.read_parquet(self.config.sequences_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Get unique prefix lengths and number of people
        prefix_lengths = sorted(df['prefix_len'].unique())
        n_people = df['person_idx'].max() + 1
        
        logger.info(f"Prefix lengths: {prefix_lengths}")
        logger.info(f"Number of people: {n_people}")
        
        all_rows = []
        
        # Process each prefix length
        for prefix_len in tqdm(prefix_lengths, desc="Computing statistics"):
            prefix_df = df[df['prefix_len'] == prefix_len]
            
            if len(prefix_df) == 0:
                continue
            
            # Compute comparison statistics (12 rows)
            comparison_rows = self._compute_comparison_stats(prefix_df, prefix_len, n_people)
            all_rows.extend(comparison_rows)
            
            # Compute token frequency statistics (V rows)
            freq_rows = self._compute_token_frequency_stats(prefix_df, prefix_len, n_people)
            all_rows.extend(freq_rows)
        
        # Convert to DataFrame
        logger.info("Building DataFrame...")
        stats_df = pd.DataFrame(all_rows)
        
        # Ensure column order for full DataFrame
        base_cols = ['prefix_len', 'row_type', 'token_id', 'token']
        person_cols = []
        for i in range(n_people):
            person_cols.extend([f'p{i}_num', f'p{i}_den'])
        total_cols = ['total_num', 'total_den']
        
        all_cols = base_cols + person_cols + total_cols
        stats_df = stats_df[all_cols]
        
        # Save full CSV (with per-person columns)
        logger.info(f"Saving full statistics: {self.config.statistics_path}")
        stats_df.to_csv(self.config.statistics_path, index=False)
        
        full_file_size = os.path.getsize(self.config.statistics_path) / (1024 * 1024)
        
        # Create and save summary CSV (without per-person columns)
        summary_cols = base_cols + total_cols
        summary_df = stats_df[summary_cols].copy()
        summary_df['rate'] = summary_df['total_num'] / summary_df['total_den'].replace(0, np.nan)
        
        logger.info(f"Saving summary statistics: {self.config.statistics_summary_path}")
        summary_df.to_csv(self.config.statistics_summary_path, index=False)
        
        summary_file_size = os.path.getsize(self.config.statistics_summary_path) / (1024 * 1024)
        
        elapsed = time.time() - start_time
        
        # Summary info
        n_comparison_rows = len([r for r in all_rows if r['row_type'] != 'token_frequency'])
        n_freq_rows = len([r for r in all_rows if r['row_type'] == 'token_frequency'])
        
        logger.info("="*60)
        logger.info("Statistics Complete!")
        logger.info(f"  Full output: {self.config.statistics_path}")
        logger.info(f"    Size: {full_file_size:.1f} MB")
        logger.info(f"    Columns: {len(stats_df.columns)} ({n_people} people × 2 + 6)")
        logger.info(f"  Summary output: {self.config.statistics_summary_path}")
        logger.info(f"    Size: {summary_file_size:.1f} MB")
        logger.info(f"    Columns: {len(summary_df.columns)}")
        logger.info(f"  Total rows: {len(all_rows)}")
        logger.info(f"    - Comparison rows: {n_comparison_rows} ({len(prefix_lengths)} prefixes × 12)")
        logger.info(f"    - Token frequency rows: {n_freq_rows}")
        logger.info(f"  PAD exclusion mode: {self.config.pad_exclusion_mode}")
        logger.info(f"  Time: {elapsed/60:.1f} minutes")
        logger.info("="*60)
        
        # Save metadata
        metadata = {
            'model_name': self.config.model_name,
            'n_people': int(n_people),
            'prefix_lengths': [int(p) for p in prefix_lengths],
            'vocab_size': self.vocab_size,
            'total_rows': len(all_rows),
            'comparison_rows': n_comparison_rows,
            'token_freq_rows': n_freq_rows,
            'pad_id': int(self.pad_id),
            'pad_exclusion_mode': self.config.pad_exclusion_mode,
            'elapsed_seconds': elapsed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        metadata_path = os.path.join(self.config.output_dir, 'statistics_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return self.config.statistics_path, self.config.statistics_summary_path


def load_config(config_path: str) -> StatsConfig:
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    output_dir = cfg['output_dir']
    
    return StatsConfig(
        model_name=cfg['model_name'],
        sequences_path=cfg['sequences_path'],
        statistics_path=cfg.get('statistics_path', os.path.join(output_dir, 'statistics_full.csv')),
        statistics_summary_path=cfg.get('statistics_summary_path', os.path.join(output_dir, 'statistics_summary.csv')),
        vocab_path=cfg['vocab_path'],
        output_dir=output_dir,
        pad_id=cfg.get('pad_id', 0),
        top_n_tokens=cfg.get('top_n_tokens', 0),
        pad_exclusion_mode=cfg.get('pad_exclusion_mode', 'both'),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistics (CPU phase)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PAD Token Exclusion Modes:
    seq1  - Exclude position only if PAD in first sequence (ground truth)
    seq2  - Exclude position only if PAD in second sequence (generated)
    both  - Exclude position if PAD in either sequence (default)
        """
    )
    parser.add_argument("--config", required=True, help="Path to run config YAML")
    args = parser.parse_args()
    
    config = load_config(args.config)
    computer = StatisticsComputer(config)
    computer.compute()


if __name__ == "__main__":
    main()
