#!/usr/bin/env python3
"""
Extract Age Information from HDF5 Files

Efficiently extracts the full age stream (input_ids[idx, 2, :]) for each person 
and saves it to a Parquet file for later use in position-dependent age statistics.

The HDF5 file has input_ids with shape (N, 4, L) where:
    - Stream 0: Token IDs
    - Stream 1: Absolute position (days from genesis)
    - Stream 2: Age (years since birth at each event)
    - Stream 3: Segment info

This script extracts stream 2 (age) as a full comma-separated string, allowing
the statistics computation to look up the age at any position (prefix_len - 1).

Usage:
    python extract_ages.py --h5_path /path/to/encoded.h5 --output /path/to/ages.parquet
    
    # Or with specific indices
    python extract_ages.py --h5_path /path/to/encoded.h5 --output /path/to/ages.parquet \
        --indices 0,1,2,3,4
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_age_at_position(age_stream: np.ndarray, position: int = 2) -> int:
    """
    Get age at a specific position in the sequence.
    
    The age stream contains the age (in years) at each token position.
    Position 0 is typically [CLS], position 1 is start of background,
    so position 2 is often the first meaningful age indicator.
    
    Args:
        age_stream: 1D array of ages for each position
        position: Position to extract age from (default 2)
    
    Returns:
        Age at the specified position
    """
    if len(age_stream) > position:
        return int(age_stream[position])
    elif len(age_stream) > 0:
        return int(age_stream[0])
    return 0


def get_decade_bucket(age: int) -> str:
    """
    Map age to decade bucket string.
    
    Args:
        age: Age in years
    
    Returns:
        Decade bucket string like "0-9", "10-19", etc.
    """
    if age < 0:
        return "unknown"
    elif age >= 100:
        return "100+"
    else:
        decade_start = (age // 10) * 10
        decade_end = decade_start + 9
        return f"{decade_start}-{decade_end}"


def extract_ages_from_h5(
    h5_path: str,
    indices: Optional[List[int]] = None,
    batch_size: int = 10000,
    pad_id: int = 0,
) -> pd.DataFrame:
    """
    Extract full age streams from HDF5 file for specified indices.
    
    Args:
        h5_path: Path to HDF5 file
        indices: List of indices to extract (None = all)
        batch_size: Batch size for processing
        pad_id: PAD token ID to find real sequence length
    
    Returns:
        DataFrame with columns: h5_idx, rinpersoon_id, age_stream, real_length
    """
    logger.info(f"Opening HDF5 file: {h5_path}")
    
    with h5py.File(h5_path, 'r', libver='latest', swmr=True) as f:
        dataset_size = f['input_ids'].shape[0]
        seq_length = f['input_ids'].shape[2]
        logger.info(f"Dataset size: {dataset_size}, sequence length: {seq_length}")
        
        # Determine which indices to process
        if indices is None:
            indices = list(range(dataset_size))
        else:
            # Filter out invalid indices
            indices = [i for i in indices if 0 <= i < dataset_size]
        
        logger.info(f"Extracting ages for {len(indices)} sequences")
        
        records = []
        
        # Process in batches for efficiency
        for batch_start in tqdm(range(0, len(indices), batch_size), desc="Extracting ages"):
            batch_end = min(batch_start + batch_size, len(indices))
            batch_indices = indices[batch_start:batch_end]
            
            for idx in batch_indices:
                # Read token stream to find real length
                token_stream = f['input_ids'][idx, 0, :]  # Shape: (L,)
                pad_positions = np.where(token_stream == pad_id)[0]
                real_length = int(pad_positions[0]) if len(pad_positions) > 0 else len(token_stream)
                
                # Read the full age stream (input_ids[idx, 2, :])
                age_stream = f['input_ids'][idx, 2, :]  # Shape: (L,)
                
                # Get sequence ID if available
                if 'sequence_id' in f:
                    rinpersoon_id = int(f['sequence_id'][idx])
                else:
                    rinpersoon_id = idx
                
                # Store the full age stream as comma-separated string
                records.append({
                    'h5_idx': idx,
                    'rinpersoon_id': rinpersoon_id,
                    'age_stream': ','.join(map(str, age_stream.tolist())),
                    'real_length': real_length,
                })
    
    df = pd.DataFrame(records)
    logger.info(f"Extracted {len(df)} age records")
    
    # Log some statistics about ages
    if len(df) > 0:
        # Sample a few ages at different positions to show distribution
        sample_ages = []
        for _, row in df.head(100).iterrows():
            ages = [int(a) for a in row['age_stream'].split(',')]
            if len(ages) > 100:
                sample_ages.append(ages[100])  # Age at position 100
        
        if sample_ages:
            logger.info(f"Sample ages at position 100: min={min(sample_ages)}, max={max(sample_ages)}, mean={np.mean(sample_ages):.1f}")
    
    return df


def extract_ages_for_generation(
    original_sequences_path: str,
    h5_path: str,
    output_path: str,
    pad_id: int = 0,
) -> str:
    """
    Extract full age streams for people in a generation run's original_sequences.parquet.
    
    This function reads the h5_idx from original_sequences.parquet and extracts
    the corresponding full age streams from the HDF5 file.
    
    Args:
        original_sequences_path: Path to original_sequences.parquet from generation
        h5_path: Path to HDF5 file with sequences
        output_path: Path to save ages parquet
        pad_id: PAD token ID to find real sequence length
    
    Returns:
        Path to saved ages parquet
    """
    logger.info(f"Loading original sequences: {original_sequences_path}")
    original_df = pd.read_parquet(original_sequences_path)
    
    # Get unique h5 indices
    h5_indices = original_df['h5_idx'].unique().tolist()
    logger.info(f"Found {len(h5_indices)} unique h5 indices")
    
    # Extract ages
    ages_df = extract_ages_from_h5(h5_path, h5_indices, pad_id=pad_id)
    
    # Merge with local_idx and is_buddy from original_sequences
    idx_mapping = original_df[['local_idx', 'h5_idx', 'is_buddy']].drop_duplicates()
    
    # Merge
    result_df = ages_df.merge(idx_mapping, on='h5_idx', how='left')
    
    # Reorder columns
    result_df = result_df[['local_idx', 'h5_idx', 'rinpersoon_id', 'age_stream', 'real_length', 'is_buddy']]
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    table = pa.Table.from_pandas(result_df)
    pq.write_table(table, output_path)
    
    logger.info(f"Saved ages to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract full age streams from HDF5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract all ages from H5 file
    python extract_ages.py --h5_path /path/to/encoded.h5 --output /path/to/ages.parquet
    
    # Extract ages for specific indices
    python extract_ages.py --h5_path /path/to/encoded.h5 --output /path/to/ages.parquet \\
        --indices "0,1,2,3,4,5"
    
    # Extract ages for a generation run
    python extract_ages.py \\
        --original_sequences /path/to/original_sequences.parquet \\
        --h5_path /path/to/encoded.h5 \\
        --output /path/to/ages.parquet

Note:
    The output contains full age streams (comma-separated) for position-dependent
    age lookups. The age at any prefix_len p is age_stream[p-1].
        """
    )
    
    parser.add_argument("--h5_path", required=True, help="Path to HDF5 file")
    parser.add_argument("--output", required=True, help="Output path for ages parquet")
    parser.add_argument("--indices", default=None, 
                        help="Comma-separated list of indices to extract (default: all)")
    parser.add_argument("--original_sequences", default=None,
                        help="Path to original_sequences.parquet (extracts only those indices)")
    parser.add_argument("--pad_id", type=int, default=0,
                        help="PAD token ID to find real sequence length (default: 0)")
    
    args = parser.parse_args()
    
    if args.original_sequences:
        # Extract for specific generation run
        extract_ages_for_generation(
            args.original_sequences,
            args.h5_path,
            args.output,
            args.pad_id,
        )
    else:
        # Extract from H5 directly
        indices = None
        if args.indices:
            indices = [int(x.strip()) for x in args.indices.split(',')]
        
        ages_df = extract_ages_from_h5(
            args.h5_path,
            indices,
            pad_id=args.pad_id,
        )
        
        # Save
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        table = pa.Table.from_pandas(ages_df)
        pq.write_table(table, args.output)
        
        logger.info(f"Saved ages to: {args.output}")


if __name__ == "__main__":
    main()
