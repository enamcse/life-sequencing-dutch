#!/usr/bin/env python3
"""
Compute Statistics (CPU Phase)

Reads generated sequences and original sequences from Parquet files and computes all statistics.
Outputs two CSV files:
    1. statistics_full.csv - Full data with per-person columns
    2. statistics_summary.csv - Aggregated data only (no per-person columns)

Input files:
    - original_sequences.parquet - Original sequences with columns:
        local_idx, h5_idx, rinpersoon_id, original_sequence, real_length, is_buddy
    - generated_sequences.parquet (sequences_path) - Generated sequences with columns:
        person_idx, rinpersoon_id, buddy_idx, buddy_rinpersoon_id, prefix_len, 
        generation_idx, generated_tokens, generated_len

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
    original_sequences_path: str  # Path to original_sequences.parquet
    statistics_path: str  # Full CSV with per-person columns
    statistics_summary_path: str  # Summary CSV without per-person columns
    vocab_path: str
    output_dir: str
    
    # Horizon (tokens to compare)
    horizon: int = 20
    
    # PAD token ID (loaded from metadata or vocab)
    pad_id: int = 0
    
    # Include all tokens or just top N
    top_n_tokens: int = 0  # 0 = all tokens
    
    # PAD token exclusion mode: 'seq1', 'seq2', or 'both'
    pad_exclusion_mode: str = 'both'
    
    # Path to ages.parquet for age-based statistics
    ages_path: Optional[str] = None
    
    # Whether to compute statistics by age group
    compute_by_age: bool = False
    
    # Output paths for by-age statistics
    statistics_by_age_path: Optional[str] = None
    statistics_by_age_summary_path: Optional[str] = None


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
            self.horizon = metadata.get('horizon', config.horizon)
            logger.info(f"PAD token ID from metadata: {self.pad_id}")
            logger.info(f"Horizon from metadata: {self.horizon}")
        else:
            self.pad_id = config.pad_id
            self.horizon = config.horizon
        
        # Load original sequences
        logger.info(f"Loading original sequences: {config.original_sequences_path}")
        self.original_df = pd.read_parquet(config.original_sequences_path)
        logger.info(f"Loaded {len(self.original_df)} original sequences")
        
        # Build lookup: local_idx -> original_sequence (as list of ints)
        self.original_sequences = {}
        self.real_lengths = {}
        for _, row in self.original_df.iterrows():
            local_idx = row['local_idx']
            self.original_sequences[local_idx] = parse_tokens(row['original_sequence'])
            self.real_lengths[local_idx] = row['real_length']
        
        # Number of persons (first n are persons, next n are buddies)
        n_persons = len(self.original_df[self.original_df['is_buddy'] == False])
        n_buddies = len(self.original_df[self.original_df['is_buddy'] == True])
        logger.info(f"Persons: {n_persons}, Buddies: {n_buddies}")
        self.n_persons = n_persons
        
        # Load ages if available and by-age statistics requested
        # Ages are stored as full age streams for position-dependent decade lookup
        self.ages_df = None
        self.person_age_streams = {}  # person_idx -> list of ages for each position
        self.decade_buckets = []  # Will be populated dynamically
        
        ages_path = config.ages_path or os.path.join(config.output_dir, 'ages.parquet')
        if config.compute_by_age and os.path.exists(ages_path):
            logger.info(f"Loading ages: {ages_path}")
            self.ages_df = pd.read_parquet(ages_path)
            
            # Build person_idx -> age_stream mapping (only for non-buddies)
            for _, row in self.ages_df[~self.ages_df['is_buddy']].iterrows():
                local_idx = row['local_idx']
                age_stream_str = row.get('age_stream', '')
                if age_stream_str and not pd.isna(age_stream_str):
                    self.person_age_streams[local_idx] = [int(a) for a in str(age_stream_str).split(',')]
                else:
                    self.person_age_streams[local_idx] = []
            
            # Standard decade buckets (we'll accumulate into these)
            self.decade_buckets = [
                "0-9", "10-19", "20-29", "30-39", "40-49", 
                "50-59", "60-69", "70-79", "80-89", "90-99", "100+"
            ]
            
            logger.info(f"Loaded {len(self.person_age_streams)} person age streams")
            logger.info(f"Decade buckets: {self.decade_buckets}")
        elif config.compute_by_age:
            logger.warning(f"Ages file not found: {ages_path}")
            logger.warning("Skipping by-age statistics computation")
    
    def _get_continuation(self, local_idx: int, prefix_len: int, horizon: int) -> List[int]:
        """Get continuation tokens from original sequence."""
        if local_idx not in self.original_sequences:
            return []
        
        seq = self.original_sequences[local_idx]
        real_len = self.real_lengths.get(local_idx, len(seq))
        
        # Check if we have enough tokens
        if prefix_len + horizon > real_len:
            # Return what we have
            return seq[prefix_len:real_len]
        
        return seq[prefix_len:prefix_len + horizon]
    
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
                buddy_idx = record.get('buddy_idx', person_idx)  # buddy's local index
                
                generated = parse_tokens(record['generated_tokens'])
                
                # Select comparison target from original sequences
                if comparison_target == 'self':
                    # Person's own continuation (person_idx is local_idx for persons)
                    target = self._get_continuation(person_idx, prefix_len, self.horizon)
                elif comparison_target == 'buddy':
                    # Buddy's continuation (buddy is at local_idx = n_persons + buddy_idx)
                    buddy_local_idx = self.n_persons + buddy_idx
                    target = self._get_continuation(buddy_local_idx, prefix_len, self.horizon)
                else:  # next
                    # Next person's continuation (circular)
                    next_idx = (person_idx + 1) % n_people
                    target = self._get_continuation(next_idx, prefix_len, self.horizon)
                
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
    
    # =========================================================================
    # By-Age Statistics Computation
    # =========================================================================
    
    def _get_real_continuation_tokens_by_decade(
        self, 
        df: pd.DataFrame, 
        n_people: int
    ) -> Dict[str, Dict[int, List[List[int]]]]:
        """
        Get real continuation tokens grouped by decade.
        
        For each generation record, we look up the real continuation tokens
        (from original_sequences) and group them by the decade of the person
        at that prefix position.
        
        Returns:
            decade -> person_idx -> list of token lists
        """
        decade_person_real_tokens = {
            bucket: {i: [] for i in range(n_people)}
            for bucket in self.decade_buckets
        }
        
        # Track which (person_idx, prefix_len) combinations we've already processed
        # to avoid duplicating real tokens across multiple generations
        seen_combinations = set()
        
        for _, record in df.iterrows():
            person_idx = record['person_idx']
            prefix_len = record['prefix_len']
            
            # Only count real tokens once per (person, prefix_len) combination
            key = (person_idx, prefix_len)
            if key in seen_combinations:
                continue
            seen_combinations.add(key)
            
            # Get the age at this prefix position
            age_at_prefix = self._get_age_at_prefix(person_idx, prefix_len)
            decade_bucket = self._get_decade_bucket_for_age(age_at_prefix)
            
            if decade_bucket not in decade_person_real_tokens:
                continue
            
            # Get real continuation tokens
            real_tokens = self._get_continuation(person_idx, prefix_len, self.horizon)
            if real_tokens:
                decade_person_real_tokens[decade_bucket][person_idx].append(real_tokens)
        
        return decade_person_real_tokens
    
    def _get_age_at_prefix(self, person_idx: int, prefix_len: int) -> int:
        """
        Get age at a specific prefix position for a person.
        
        For a given person and prefix_len k, the age is the age at position k-1
        (the last token before generation starts). This is because we generate
        tokens starting at position k, so the age context is from position k-1.
        
        Args:
            person_idx: Person index (local_idx for non-buddy)
            prefix_len: Prefix length (the position where generation starts)
        
        Returns:
            Age at position prefix_len - 1, or 0 if not available
        """
        if person_idx not in self.person_age_streams:
            return 0
        
        age_stream = self.person_age_streams[person_idx]
        
        # We want the age at the last token of the prefix (position prefix_len - 1)
        age_position = prefix_len - 1
        
        if age_position < 0:
            return 0
        elif age_position < len(age_stream):
            return age_stream[age_position]
        elif len(age_stream) > 0:
            # If prefix_len exceeds stream length, use last available age
            return age_stream[-1]
        return 0
    
    def _get_decade_bucket_for_age(self, age: int) -> str:
        """Map age to decade bucket string."""
        if age < 0:
            return "unknown"
        elif age >= 100:
            return "100+"
        else:
            decade_start = (age // 10) * 10
            decade_end = decade_start + 9
            return f"{decade_start}-{decade_end}"
    
    def _compute_by_age_statistics(self, df: pd.DataFrame, prefix_lengths: List[int], n_people: int) -> Tuple[str, str]:
        """
        Compute statistics grouped by age decade and save to CSV.
        
        This aggregates statistics ACROSS all prefix_lens into decade buckets.
        The first column is 'decade' (e.g., "0-9", "10-19"), not 'prefix_len'.
        
        For each generation record:
        - Look up the age at position (prefix_len - 1) for that person
        - Map the age to a decade bucket
        - Accumulate that record's statistics into the decade bucket
        
        This gives us a view of model performance grouped by life stage,
        allowing us to see how well the model predicts tokens for different age groups.
        
        Returns:
            Tuple of (full_path, summary_path)
        """
        if self.ages_df is None or not self.decade_buckets:
            logger.warning("Ages not loaded, skipping by-age statistics")
            return None, None
        
        logger.info("="*60)
        logger.info("Computing By-Age Statistics")
        logger.info(f"  Decade buckets: {self.decade_buckets}")
        logger.info(f"  Aggregating across {len(prefix_lengths)} prefix lengths")
        logger.info("="*60)
        
        # =====================================================================
        # Step 1: Compute comparison statistics by decade
        # =====================================================================
        comparison_rows = []
        
        for row_type in self.ROW_TYPES:
            # Parse row type
            parts = row_type.split('_')
            is_ordered = parts[0] == 'ordered'
            comparison_target = parts[1]
            include_pad = parts[-1] == 'pad' and parts[-2] == 'with'
            exclude_pad = not include_pad
            
            # Initialize accumulators: decade -> person_idx -> {num, den}
            decade_person_stats = {
                bucket: {i: {'num': 0, 'den': 0} for i in range(n_people)}
                for bucket in self.decade_buckets
            }
            
            # Process ALL records across all prefix lengths
            for _, record in tqdm(df.iterrows(), total=len(df), 
                                  desc=f"By-age {row_type}", leave=False):
                person_idx = record['person_idx']
                buddy_idx = record.get('buddy_idx', person_idx)
                prefix_len = record['prefix_len']
                
                # Get the age at this specific prefix position for this person
                age_at_prefix = self._get_age_at_prefix(person_idx, prefix_len)
                decade_bucket = self._get_decade_bucket_for_age(age_at_prefix)
                
                # Skip unknown decades
                if decade_bucket not in decade_person_stats:
                    continue
                
                generated = parse_tokens(record['generated_tokens'])
                
                # Select comparison target from original sequences
                if comparison_target == 'self':
                    target = self._get_continuation(person_idx, prefix_len, self.horizon)
                elif comparison_target == 'buddy':
                    buddy_local_idx = self.n_persons + buddy_idx
                    target = self._get_continuation(buddy_local_idx, prefix_len, self.horizon)
                else:  # next
                    next_idx = (person_idx + 1) % n_people
                    target = self._get_continuation(next_idx, prefix_len, self.horizon)
                
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
                
                decade_person_stats[decade_bucket][person_idx]['num'] += num
                decade_person_stats[decade_bucket][person_idx]['den'] += den
            
            # Build one row per decade for this row_type
            for decade_bucket in self.decade_buckets:
                row = {
                    'decade': decade_bucket,
                    'row_type': row_type,
                    'token_id': None,
                    'token': None,
                }
                
                total_num = 0
                total_den = 0
                
                for i in range(n_people):
                    row[f'p{i}_num'] = decade_person_stats[decade_bucket][i]['num']
                    row[f'p{i}_den'] = decade_person_stats[decade_bucket][i]['den']
                    total_num += decade_person_stats[decade_bucket][i]['num']
                    total_den += decade_person_stats[decade_bucket][i]['den']
                
                row['total_num'] = total_num
                row['total_den'] = total_den
                
                comparison_rows.append(row)
        
        # =====================================================================
        # Step 2: Compute token frequency statistics by decade
        # =====================================================================
        logger.info("Computing token frequencies by decade...")
        
        # Collect generated tokens: decade -> person_idx -> list of token lists
        decade_person_tokens = {
            bucket: {i: [] for i in range(n_people)}
            for bucket in self.decade_buckets
        }
        
        for _, record in tqdm(df.iterrows(), total=len(df), 
                              desc="By-age token freq", leave=False):
            person_idx = record['person_idx']
            prefix_len = record['prefix_len']
            
            # Get the age at this specific prefix position for this person
            age_at_prefix = self._get_age_at_prefix(person_idx, prefix_len)
            decade_bucket = self._get_decade_bucket_for_age(age_at_prefix)
            
            if decade_bucket not in decade_person_tokens:
                continue
            
            generated = parse_tokens(record['generated_tokens'])
            decade_person_tokens[decade_bucket][person_idx].append(generated)
        
        # Compute frequencies per decade and person
        decade_person_freqs = {}
        decade_person_totals = {}
        
        for bucket in self.decade_buckets:
            decade_person_freqs[bucket] = {}
            decade_person_totals[bucket] = {}
            for i in range(n_people):
                freq, total = compute_token_frequencies(decade_person_tokens[bucket][i])
                decade_person_freqs[bucket][i] = freq
                decade_person_totals[bucket][i] = total
        
        # Determine which tokens to include
        if self.config.top_n_tokens > 0:
            all_freqs = Counter()
            for bucket in self.decade_buckets:
                for i in range(n_people):
                    all_freqs.update(decade_person_freqs[bucket][i])
            top_tokens = [tid for tid, _ in all_freqs.most_common(self.config.top_n_tokens)]
        else:
            top_tokens = self.all_token_ids
        
        # Build frequency rows: one row per (decade, token_id)
        freq_rows = []
        for decade_bucket in self.decade_buckets:
            for token_id in top_tokens:
                row = {
                    'decade': decade_bucket,
                    'row_type': 'token_frequency',
                    'token_id': token_id,
                    'token': self.id_to_token.get(token_id, f'UNK_{token_id}'),
                }
                
                total_num = 0
                total_den = 0
                
                for i in range(n_people):
                    num = decade_person_freqs[decade_bucket][i].get(token_id, 0)
                    den = decade_person_totals[decade_bucket][i]
                    row[f'p{i}_num'] = num
                    row[f'p{i}_den'] = den
                    total_num += num
                    total_den += den
                
                row['total_num'] = total_num
                row['total_den'] = total_den
                
                freq_rows.append(row)
        
        # =====================================================================
        # Step 3: Combine and save
        # =====================================================================
        all_rows = comparison_rows + freq_rows
        
        logger.info("Building by-age DataFrame...")
        stats_df = pd.DataFrame(all_rows)
        
        # Ensure column order: decade, row_type, token_id, token, p0_num, p0_den, ..., total_num, total_den
        base_cols = ['decade', 'row_type', 'token_id', 'token']
        person_cols = []
        for i in range(n_people):
            person_cols.extend([f'p{i}_num', f'p{i}_den'])
        total_cols = ['total_num', 'total_den']
        
        all_cols = base_cols + person_cols + total_cols
        
        # Ensure all columns exist
        for col in all_cols:
            if col not in stats_df.columns:
                stats_df[col] = 0
        
        stats_df = stats_df[all_cols]
        
        # Sort by decade order, then row_type
        decade_order = {bucket: i for i, bucket in enumerate(self.decade_buckets)}
        stats_df['_decade_order'] = stats_df['decade'].map(decade_order)
        stats_df = stats_df.sort_values(['_decade_order', 'row_type', 'token_id']).drop(columns=['_decade_order'])
        
        # Determine output paths
        output_dir = self.config.output_dir
        n_people_val = n_people
        n_generations = len(df) // (len(prefix_lengths) * n_people) if n_people > 0 else 0
        
        # Default paths with n, c indicators
        if self.config.statistics_by_age_path:
            by_age_full_path = self.config.statistics_by_age_path
        else:
            by_age_full_path = os.path.join(output_dir, f'statistics_by_age_n{n_people_val}_c{n_generations}_full.csv')
        
        if self.config.statistics_by_age_summary_path:
            by_age_summary_path = self.config.statistics_by_age_summary_path
        else:
            by_age_summary_path = os.path.join(output_dir, f'statistics_by_age_n{n_people_val}_c{n_generations}_summary.csv')
        
        # Save full CSV
        logger.info(f"Saving by-age full statistics: {by_age_full_path}")
        stats_df.to_csv(by_age_full_path, index=False)
        
        full_file_size = os.path.getsize(by_age_full_path) / (1024 * 1024)
        
        # Create and save summary CSV (without per-person columns)
        summary_cols = base_cols + total_cols
        summary_df = stats_df[summary_cols].copy()
        summary_df['rate'] = summary_df['total_num'] / summary_df['total_den'].replace(0, np.nan)
        
        logger.info(f"Saving by-age summary statistics: {by_age_summary_path}")
        summary_df.to_csv(by_age_summary_path, index=False)
        
        summary_file_size = os.path.getsize(by_age_summary_path) / (1024 * 1024)
        
        # Log summary
        n_comparison_rows = len(comparison_rows)
        n_freq_rows = len(freq_rows)
        
        logger.info(f"By-Age Statistics Complete!")
        logger.info(f"  Full output: {by_age_full_path} ({full_file_size:.1f} MB)")
        logger.info(f"    Columns: {len(stats_df.columns)} ({n_people} people × 2 + 6)")
        logger.info(f"    Rows per decade: {12 + len(top_tokens)} (12 comparisons + {len(top_tokens)} tokens)")
        logger.info(f"  Summary output: {by_age_summary_path} ({summary_file_size:.1f} MB)")
        logger.info(f"  Total rows: {len(all_rows)} ({len(self.decade_buckets)} decades)")
        logger.info(f"    - Comparison rows: {n_comparison_rows} ({len(self.decade_buckets)} decades × 12)")
        logger.info(f"    - Token frequency rows: {n_freq_rows} ({len(self.decade_buckets)} decades × {len(top_tokens)} tokens)")
        
        # =====================================================================
        # Step 4: Token Counts Spreadsheet with N_d and Real/Simulated Counts
        # =====================================================================
        self._compute_token_counts_spreadsheet(
            df, decade_person_tokens, n_people, n_generations, output_dir
        )
        
        # =====================================================================
        # Step 5: Age Progression Spreadsheet
        # =====================================================================
        self._compute_age_progression_spreadsheet(
            df, n_people, n_generations, output_dir
        )
        
        return by_age_full_path, by_age_summary_path
    
    def _compute_token_counts_spreadsheet(
        self,
        df: pd.DataFrame,
        decade_person_simulated_tokens: Dict[str, Dict[int, List[List[int]]]],
        n_people: int,
        n_generations: int,
        output_dir: str
    ):
        """
        Compute and save a token counts spreadsheet with:
        - N_d: Number of (person, prefix_len) combinations contributing to each decade
        - unique_people: Number of distinct people contributing to each decade
        - For each token: simulated_count, real_count
        - Expected totals: real = N_d × horizon, simulated = N_d × horizon × n_generations
        
        Output format:
            decade, N_d, unique_people, token_id, token, simulated_count, real_count
        
        This provides a clean view of raw token counts for sanity checking.
        
        Note: N_d counts (person, prefix_len) pairs, so one person can contribute
        multiple times to the same decade at different prefix positions. The sum
        of unique_people across decades may exceed n_people since a person can
        appear in multiple decades as they age through life.
        """
        logger.info("="*60)
        logger.info("Computing Token Counts Spreadsheet")
        logger.info("="*60)
        
        # Get real continuation tokens by decade
        decade_person_real_tokens = self._get_real_continuation_tokens_by_decade(df, n_people)
        
        # Compute N_d: number of unique (person, prefix) combinations per decade
        # Also compute unique_people: number of distinct people per decade
        decade_n_d = {}
        decade_unique_people = {}
        seen_combinations_per_decade = {bucket: set() for bucket in self.decade_buckets}
        seen_people_per_decade = {bucket: set() for bucket in self.decade_buckets}
        
        for _, record in df.iterrows():
            person_idx = record['person_idx']
            prefix_len = record['prefix_len']
            
            age_at_prefix = self._get_age_at_prefix(person_idx, prefix_len)
            decade_bucket = self._get_decade_bucket_for_age(age_at_prefix)
            
            if decade_bucket in seen_combinations_per_decade:
                seen_combinations_per_decade[decade_bucket].add((person_idx, prefix_len))
                seen_people_per_decade[decade_bucket].add(person_idx)
        
        for bucket in self.decade_buckets:
            decade_n_d[bucket] = len(seen_combinations_per_decade[bucket])
            decade_unique_people[bucket] = len(seen_people_per_decade[bucket])
        
        # Compute token frequencies for simulated and real
        rows = []
        
        for decade_bucket in self.decade_buckets:
            n_d = decade_n_d[decade_bucket]
            unique_ppl = decade_unique_people[decade_bucket]
            
            if n_d == 0:
                continue
            
            # Aggregate simulated tokens across all people for this decade
            simulated_counter = Counter()
            total_simulated_tokens = 0
            for person_idx in range(n_people):
                for token_list in decade_person_simulated_tokens[decade_bucket][person_idx]:
                    simulated_counter.update(token_list)
                    total_simulated_tokens += len(token_list)
            
            # Aggregate real tokens across all people for this decade
            real_counter = Counter()
            total_real_tokens = 0
            for person_idx in range(n_people):
                for token_list in decade_person_real_tokens[decade_bucket][person_idx]:
                    real_counter.update(token_list)
                    total_real_tokens += len(token_list)
            
            # Get all tokens that appear in either simulated or real
            all_tokens = set(simulated_counter.keys()) | set(real_counter.keys())
            
            # Build rows for this decade
            for token_id in sorted(all_tokens):
                rows.append({
                    'decade': decade_bucket,
                    'N_d': n_d,
                    'unique_people': unique_ppl,
                    'token_id': token_id,
                    'token': self.id_to_token.get(token_id, f'UNK_{token_id}'),
                    'simulated_count': simulated_counter.get(token_id, 0),
                    'real_count': real_counter.get(token_id, 0),
                })
            
            # Log expected vs actual totals for this decade
            expected_real = n_d * self.horizon
            expected_simulated = n_d * self.horizon * n_generations
            logger.info(f"  Decade {decade_bucket}: N_d={n_d}, unique_people={unique_ppl}")
            logger.info(f"    Real tokens: {total_real_tokens} (expected: {expected_real})")
            logger.info(f"    Simulated tokens: {total_simulated_tokens} (expected: {expected_simulated})")
        
        # Create DataFrame
        counts_df = pd.DataFrame(rows)
        
        # Save to CSV
        counts_path = os.path.join(output_dir, f'token_counts_by_decade_n{n_people}_c{n_generations}.csv')
        logger.info(f"Saving token counts spreadsheet: {counts_path}")
        counts_df.to_csv(counts_path, index=False)
        
        file_size = os.path.getsize(counts_path) / (1024 * 1024)
        logger.info(f"  Size: {file_size:.1f} MB")
        logger.info(f"  Rows: {len(counts_df)}")
        logger.info(f"  Decades with data: {len([d for d in self.decade_buckets if decade_n_d.get(d, 0) > 0])}")
        
        # Also save a summary with just decade-level info
        decade_summary_rows = []
        for decade_bucket in self.decade_buckets:
            n_d = decade_n_d[decade_bucket]
            unique_ppl = decade_unique_people[decade_bucket]
            if n_d == 0:
                continue
            
            # Sum up all tokens for this decade
            decade_rows = counts_df[counts_df['decade'] == decade_bucket]
            total_simulated = decade_rows['simulated_count'].sum()
            total_real = decade_rows['real_count'].sum()
            
            decade_summary_rows.append({
                'decade': decade_bucket,
                'N_d': n_d,
                'unique_people': unique_ppl,
                'total_real_tokens': total_real,
                'total_simulated_tokens': total_simulated,
                'expected_real_tokens': n_d * self.horizon,
                'expected_simulated_tokens': n_d * self.horizon * n_generations,
                'unique_real_tokens': len(decade_rows[decade_rows['real_count'] > 0]),
                'unique_simulated_tokens': len(decade_rows[decade_rows['simulated_count'] > 0]),
            })
        
        decade_summary_df = pd.DataFrame(decade_summary_rows)
        summary_path = os.path.join(output_dir, f'decade_summary_n{n_people}_c{n_generations}.csv')
        logger.info(f"Saving decade summary: {summary_path}")
        decade_summary_df.to_csv(summary_path, index=False)
        
        # Log total unique people check
        total_unique = sum(decade_unique_people.values())
        logger.info(f"  Total unique_people across decades: {total_unique} (may exceed n={n_people} since people age through multiple decades)")
        
        logger.info("Token Counts Spreadsheet Complete!")
        
        return counts_path, summary_path
    
    def _compute_age_progression_spreadsheet(
        self,
        df: pd.DataFrame,
        n_people: int,
        n_generations: int,
        output_dir: str
    ):
        """
        Compute and save an age progression spreadsheet showing which decade
        each person falls into at each prefix_len.
        
        Output format:
            prefix_len, p0, p1, p2, ..., p{n-1}
            1, 0-9, 0-9, 0-9, ...
            101, 0-9, 30-39, 10-19, ...
            201, 0-9, 30-39, 20-29, ...
            ...
        
        This provides a clear view of how age progression varies across people,
        useful for sanity checking the age data and understanding the cohort.
        """
        logger.info("="*60)
        logger.info("Computing Age Progression Spreadsheet")
        logger.info("="*60)
        
        # Get unique prefix lengths
        prefix_lengths = sorted(df['prefix_len'].unique())
        
        # Build the spreadsheet: prefix_len -> person_idx -> decade
        rows = []
        
        for prefix_len in prefix_lengths:
            row = {'prefix_len': prefix_len}
            
            for person_idx in range(n_people):
                age = self._get_age_at_prefix(person_idx, prefix_len)
                decade = self._get_decade_bucket_for_age(age)
                row[f'p{person_idx}'] = decade
            
            rows.append(row)
        
        # Create DataFrame
        progression_df = pd.DataFrame(rows)
        
        # Ensure column order: prefix_len, p0, p1, p2, ...
        cols = ['prefix_len'] + [f'p{i}' for i in range(n_people)]
        progression_df = progression_df[cols]
        
        # Save to CSV
        progression_path = os.path.join(output_dir, f'age_progression_n{n_people}_c{n_generations}.csv')
        logger.info(f"Saving age progression spreadsheet: {progression_path}")
        progression_df.to_csv(progression_path, index=False)
        
        file_size = os.path.getsize(progression_path) / (1024 * 1024)
        logger.info(f"  Size: {file_size:.1f} MB")
        logger.info(f"  Rows (prefix_lens): {len(progression_df)}")
        logger.info(f"  Columns (people): {n_people}")
        
        # Log some statistics about age progression
        # Count how many decades each person spans
        decades_per_person = []
        for person_idx in range(n_people):
            col = f'p{person_idx}'
            unique_decades = progression_df[col].nunique()
            decades_per_person.append(unique_decades)
        
        avg_decades = np.mean(decades_per_person)
        min_decades = min(decades_per_person)
        max_decades = max(decades_per_person)
        logger.info(f"  Decades spanned per person: min={min_decades}, avg={avg_decades:.1f}, max={max_decades}")
        
        logger.info("Age Progression Spreadsheet Complete!")
        
        return progression_path

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
        
        # Compute n_generations from data
        n_generations = len(df) // (len(prefix_lengths) * n_people) if (n_people > 0 and len(prefix_lengths) > 0) else 0
        
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
            'n_generations': int(n_generations),
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
        
        # Compute by-age statistics if enabled and ages are available
        if self.config.compute_by_age and self.ages_df is not None and len(self.decade_buckets) > 0:
            logger.info("")
            self._compute_by_age_statistics(df, prefix_lengths, n_people)
        
        return self.config.statistics_path, self.config.statistics_summary_path


def load_config(config_path: str) -> StatsConfig:
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    output_dir = cfg['output_dir']
    
    return StatsConfig(
        model_name=cfg['model_name'],
        sequences_path=cfg['sequences_path'],
        original_sequences_path=cfg.get('original_sequences_path', 
                                        os.path.join(output_dir, 'original_sequences.parquet')),
        statistics_path=cfg.get('statistics_path', os.path.join(output_dir, 'statistics_full.csv')),
        statistics_summary_path=cfg.get('statistics_summary_path', os.path.join(output_dir, 'statistics_summary.csv')),
        vocab_path=cfg['vocab_path'],
        output_dir=output_dir,
        horizon=cfg.get('horizon', 20),
        pad_id=cfg.get('pad_id', 0),
        top_n_tokens=cfg.get('top_n_tokens', 0),
        pad_exclusion_mode=cfg.get('pad_exclusion_mode', 'both'),
        ages_path=cfg.get('ages_path'),
        compute_by_age=cfg.get('compute_by_age', False),
        statistics_by_age_path=cfg.get('statistics_by_age_path'),
        statistics_by_age_summary_path=cfg.get('statistics_by_age_summary_path'),
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
