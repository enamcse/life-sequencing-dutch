#!/usr/bin/env python3
"""
H5 Sequence Extractor - Extract sequences matching specific criteria from large HDF5 datasets.

Extracts sequences that meet ALL of the following criteria:
1. Childhood start: age at index 6 is in 0-9
2. Full length: token at index 1023 is non-zero OR age at index 1023 is non-zero
3. End-of-life: age at index 1023 is in 70-79, 80-89, or 90-99

OR use a config file with position-based age criteria:
    {
        "position_age_criteria": [
            {"position": 6, "age_min": 0, "age_max": 9},
            {"position": 100, "age_min": 10, "age_max": 19},
            {"position": 200, "age_min": 20, "age_max": 29},
            ...
        ]
    }

Usage:
    python h5_sequence_extractor.py --h5_file encoded.h5 --output extracted.h5 --n_sequences 10000
    python h5_sequence_extractor.py --h5_file encoded.h5 --output extracted.h5 --n_sequences 10000 --n_workers 16
    python h5_sequence_extractor.py --h5_file encoded.h5 --output extracted.h5 --criteria childhood_start,full_length
    
    # Use config file for position-based age criteria
    python h5_sequence_extractor.py --h5_file encoded.h5 --output extracted.h5 --config criteria_config.json

The HDF5 file should have 'input_ids' with shape (N, 4, 1024):
    - input_ids[:, 0, :] = token IDs
    - input_ids[:, 2, :] = ages

Performance optimizations:
    - Vectorized numpy operations (no Python loops)
    - Large batch processing (default 500K sequences per batch)
    - Multi-process parallel chunk processing
    - Efficient batch extraction with sorted indices
    - tqdm progress bars for tracking
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Set, Any

import h5py
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# =============================================================================
# Config Loading and Validation
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate a criteria config file (JSON or YAML)."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            try:
                import yaml
                config = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files. Install with: pip install pyyaml")
        else:
            config = json.load(f)
    
    # Validate config
    if 'position_age_criteria' not in config:
        raise ValueError("Config must contain 'position_age_criteria' list")
    
    for i, criterion in enumerate(config['position_age_criteria']):
        required = ['position', 'age_min', 'age_max']
        for key in required:
            if key not in criterion:
                raise ValueError(f"Criterion {i} missing required key: {key}")
        
        if not isinstance(criterion['position'], int) or criterion['position'] < 0:
            raise ValueError(f"Criterion {i}: position must be a non-negative integer")
        if not isinstance(criterion['age_min'], int):
            raise ValueError(f"Criterion {i}: age_min must be an integer")
        if not isinstance(criterion['age_max'], int):
            raise ValueError(f"Criterion {i}: age_max must be an integer")
        if criterion['age_min'] > criterion['age_max']:
            raise ValueError(f"Criterion {i}: age_min ({criterion['age_min']}) > age_max ({criterion['age_max']})")
    
    return config


def create_default_lifespan_config() -> Dict[str, Any]:
    """Create the default lifespan config: childhood at 6, then decade progression."""
    config = {
        "name": "lifespan_decade_progression",
        "description": "Sequences with childhood start and decade progression through life",
        "position_age_criteria": [
            {"position": 6, "age_min": 0, "age_max": 9, "label": "childhood"},
            {"position": 100, "age_min": 10, "age_max": 19, "label": "teens"},
            {"position": 200, "age_min": 20, "age_max": 29, "label": "twenties"},
            {"position": 300, "age_min": 30, "age_max": 39, "label": "thirties"},
            {"position": 400, "age_min": 40, "age_max": 49, "label": "forties"},
            {"position": 500, "age_min": 50, "age_max": 59, "label": "fifties"},
            {"position": 600, "age_min": 60, "age_max": 69, "label": "sixties"},
            {"position": 700, "age_min": 70, "age_max": 79, "label": "seventies"},
            {"position": 800, "age_min": 80, "age_max": 89, "label": "eighties"},
            {"position": 900, "age_min": 90, "age_max": 99, "label": "nineties"},
            {"position": 1000, "age_min": 90, "age_max": 99, "label": "nineties_end"},
        ]
    }
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save config to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to: {output_path}")


# =============================================================================
# Criteria Functions (Legacy)
# =============================================================================

def check_childhood_start(age_6: int) -> bool:
    """Check if age at index 6 is in 0-9 (childhood)."""
    return 0 <= age_6 <= 9


def check_full_length(token_1023: int, age_1023: int) -> bool:
    """Check if sequence is full length (non-zero at index 1023)."""
    return token_1023 != 0 or age_1023 != 0


def check_end_of_life(age_1023: int) -> bool:
    """Check if age at index 1023 is in 70-99."""
    return 70 <= age_1023 <= 99


def check_decade_70(age_1023: int) -> bool:
    """Check if age at index 1023 is in 70-79."""
    return 70 <= age_1023 <= 79


def check_decade_80(age_1023: int) -> bool:
    """Check if age at index 1023 is in 80-89."""
    return 80 <= age_1023 <= 89


def check_decade_90(age_1023: int) -> bool:
    """Check if age at index 1023 is in 90-99."""
    return 90 <= age_1023 <= 99


def check_all_criteria(age_6: int, token_1023: int, age_1023: int) -> bool:
    """Check if sequence meets all three criteria."""
    return (check_childhood_start(age_6) and 
            check_full_length(token_1023, age_1023) and 
            check_end_of_life(age_1023))


# =============================================================================
# Chunk Processing - Vectorized
# =============================================================================

def find_matching_indices_chunk_vectorized(
    h5_path: str, 
    start_idx: int, 
    end_idx: int,
    criteria: Set[str]
) -> np.ndarray:
    """
    Find indices of sequences matching criteria in a chunk using vectorized operations.
    Memory-efficient: only reads specific columns needed for criteria checking.
    
    Args:
        h5_path: Path to HDF5 file
        start_idx: Start index (inclusive)
        end_idx: End index (exclusive)
        criteria: Set of criteria names to check
    
    Returns:
        numpy array of global indices that match all criteria
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            input_ids = f['input_ids']
            n_sequences = end_idx - start_idx
            
            # Memory optimization: Only read the specific positions we need
            # instead of reading the entire (N, 4, 1024) array
            age_6 = input_ids[start_idx:end_idx, 2, 6].astype(np.int32)
            age_1023 = input_ids[start_idx:end_idx, 2, 1023].astype(np.int32)
            token_1023 = input_ids[start_idx:end_idx, 0, 1023].astype(np.int32)
            
            # Start with all True mask
            mask = np.ones(n_sequences, dtype=bool)
            
            # Apply each criterion using vectorized operations
            if 'childhood_start' in criteria or 'all' in criteria:
                mask &= (age_6 >= 0) & (age_6 <= 9)
            
            if 'full_length' in criteria or 'all' in criteria:
                mask &= (token_1023 != 0) | (age_1023 != 0)
            
            if 'end_of_life' in criteria or 'all' in criteria:
                mask &= (age_1023 >= 70) & (age_1023 <= 99)
            
            if 'decade_70' in criteria:
                mask &= (age_1023 >= 70) & (age_1023 <= 79)
            
            if 'decade_80' in criteria:
                mask &= (age_1023 >= 80) & (age_1023 <= 89)
            
            if 'decade_90' in criteria:
                mask &= (age_1023 >= 90) & (age_1023 <= 99)
            
            # Get matching local indices and convert to global
            local_indices = np.where(mask)[0]
            global_indices = local_indices + start_idx
            
        return global_indices
        
    except Exception as e:
        logger.error(f"Error in chunk {start_idx}-{end_idx}: {e}")
        return np.array([], dtype=np.int64)


def find_matching_indices_chunk_config(
    h5_path: str, 
    start_idx: int, 
    end_idx: int,
    position_criteria: List[Dict]
) -> np.ndarray:
    """
    Find indices of sequences matching position-based age criteria using vectorized operations.
    
    Args:
        h5_path: Path to HDF5 file
        start_idx: Start index (inclusive)
        end_idx: End index (exclusive)
        position_criteria: List of dicts with 'position', 'age_min', 'age_max'
    
    Returns:
        numpy array of global indices that match ALL criteria
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            input_ids = f['input_ids']
            n_sequences = end_idx - start_idx
            
            # Start with all True mask
            mask = np.ones(n_sequences, dtype=bool)
            
            # Apply each position-based criterion
            for criterion in position_criteria:
                pos = criterion['position']
                age_min = criterion['age_min']
                age_max = criterion['age_max']
                
                # Read age at this position
                age_at_pos = input_ids[start_idx:end_idx, 2, pos].astype(np.int32)
                
                # Apply criterion
                mask &= (age_at_pos >= age_min) & (age_at_pos <= age_max)
                
                # Early exit if no matches left
                if not mask.any():
                    return np.array([], dtype=np.int64)
            
            # Get matching local indices and convert to global
            local_indices = np.where(mask)[0]
            global_indices = local_indices + start_idx
            
        return global_indices
        
    except Exception as e:
        logger.error(f"Error in chunk {start_idx}-{end_idx}: {e}")
        return np.array([], dtype=np.int64)


# =============================================================================
# Main Extraction Functions
# =============================================================================

def find_all_matching_indices(
    h5_path: str,
    n_workers: int = 8,
    chunk_size: int = 100000,
    criteria: Optional[Set[str]] = None,
    position_criteria: Optional[List[Dict]] = None,
    max_indices: Optional[int] = None,
    sequential: bool = False
) -> np.ndarray:
    """
    Find all indices of sequences matching criteria using parallel processing.
    
    Args:
        h5_path: Path to HDF5 file
        n_workers: Number of parallel workers
        chunk_size: Chunk size for processing
        criteria: Set of criteria names (default: all three criteria) - legacy mode
        position_criteria: List of position-based age criteria from config - new mode
        max_indices: Stop early if this many indices are found
        sequential: Use sequential processing (slower but memory-safe)
    
    Returns:
        numpy array of matching indices
    """
    use_config_mode = position_criteria is not None
    
    if not use_config_mode and criteria is None:
        criteria = {'childhood_start', 'full_length', 'end_of_life'}
    
    # Get dataset size
    with h5py.File(h5_path, 'r') as f:
        n_sequences = f['input_ids'].shape[0]
        seq_length = f['input_ids'].shape[2]
        logger.info(f"Total sequences in file: {n_sequences:,}")
        logger.info(f"Sequence length: {seq_length}")
    
    # Validate position criteria
    if use_config_mode:
        for criterion in position_criteria:
            if criterion['position'] >= seq_length:
                raise ValueError(f"Position {criterion['position']} >= sequence length {seq_length}")
    
    # Create chunks
    chunks = []
    for start in range(0, n_sequences, chunk_size):
        end = min(start + chunk_size, n_sequences)
        chunks.append((start, end))
    
    logger.info(f"Processing {len(chunks)} chunks of ~{chunk_size:,} sequences each")
    if use_config_mode:
        logger.info(f"Using config-based criteria: {len(position_criteria)} position checks")
        for c in position_criteria:
            label = c.get('label', f"pos_{c['position']}")
            logger.info(f"  - {label}: position {c['position']}, age {c['age_min']}-{c['age_max']}")
    else:
        logger.info(f"Criteria: {criteria}")
    
    start_time = time.time()
    all_indices_list = []
    total_found = 0
    
    # Choose the appropriate chunk function
    if use_config_mode:
        from functools import partial
        chunk_func = partial(find_matching_indices_chunk_config, position_criteria=position_criteria)
    else:
        from functools import partial
        chunk_func = partial(find_matching_indices_chunk_vectorized, criteria=criteria)
    
    if sequential:
        # Sequential processing - memory safe
        logger.info("Using sequential processing (memory-safe mode)")
        
        for start, end in tqdm(chunks, desc="Scanning chunks", unit="chunk"):
            try:
                if use_config_mode:
                    indices = find_matching_indices_chunk_config(h5_path, start, end, position_criteria)
                else:
                    indices = find_matching_indices_chunk_vectorized(h5_path, start, end, criteria)
                
                if len(indices) > 0:
                    all_indices_list.append(indices)
                    total_found += len(indices)
                
                if max_indices is not None and total_found >= max_indices:
                    logger.info(f"Found enough indices ({total_found:,} >= {max_indices:,})")
                    break
            except Exception as e:
                logger.error(f"Error processing chunk {start}-{end}: {e}")
    else:
        # Parallel processing
        logger.info(f"Using {n_workers} parallel workers")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            if use_config_mode:
                futures = {
                    executor.submit(find_matching_indices_chunk_config, h5_path, start, end, position_criteria): (start, end)
                    for start, end in chunks
                }
            else:
                futures = {
                    executor.submit(find_matching_indices_chunk_vectorized, h5_path, start, end, criteria): (start, end)
                    for start, end in chunks
                }
            
            with tqdm(total=len(chunks), desc="Scanning chunks", unit="chunk") as pbar:
                for future in as_completed(futures):
                    start, end = futures[future]
                    try:
                        indices = future.result()
                        if len(indices) > 0:
                            all_indices_list.append(indices)
                            total_found += len(indices)
                        
                        pbar.update(1)
                        pbar.set_postfix({'found': f"{total_found:,}"})
                        
                        # Check for early stopping
                        if max_indices is not None and total_found >= max_indices:
                            logger.info(f"Found enough indices ({total_found:,} >= {max_indices:,})")
                            break
                            
                    except Exception as e:
                        logger.error(f"Error processing chunk {start}-{end}: {e}")
                        pbar.update(1)
    
    # Concatenate all indices
    if all_indices_list:
        all_indices = np.concatenate(all_indices_list)
    else:
        all_indices = np.array([], dtype=np.int64)
    
    elapsed = time.time() - start_time
    logger.info(f"Found {len(all_indices):,} matching indices in {elapsed:.1f}s ({n_sequences/elapsed:,.0f} seq/s)")
    
    return all_indices


def extract_sequences(
    h5_path: str,
    output_path: str,
    indices: np.ndarray,
    n_sequences: int = 10000,
    seed: int = 42
) -> int:
    """
    Extract sequences at given indices to a new HDF5 file.
    
    Uses batch processing for efficient I/O.
    
    Args:
        h5_path: Source HDF5 file
        output_path: Output HDF5 file
        indices: Array of indices to sample from
        n_sequences: Number of sequences to extract
        seed: Random seed for sampling
    
    Returns:
        Number of sequences actually extracted
    """
    if len(indices) < n_sequences:
        logger.warning(
            f"Only {len(indices):,} matching sequences found, "
            f"extracting all of them instead of {n_sequences:,}"
        )
        n_sequences = len(indices)
    
    # Random sample
    np.random.seed(seed)
    selected_indices = np.random.choice(indices, size=n_sequences, replace=False)
    selected_indices = np.sort(selected_indices)  # Sort for efficient sequential read
    
    logger.info(f"Extracting {n_sequences:,} sequences to {output_path}")
    
    start_time = time.time()
    
    with h5py.File(h5_path, 'r') as f_in:
        input_ids_src = f_in['input_ids']
        shape = input_ids_src.shape
        dtype = input_ids_src.dtype
        
        logger.info(f"Source shape: {shape}, dtype: {dtype}")
        
        # Create output file
        with h5py.File(output_path, 'w') as f_out:
            # Create dataset
            output_shape = (n_sequences, shape[1], shape[2])
            input_ids_dst = f_out.create_dataset(
                'input_ids', 
                shape=output_shape, 
                dtype=dtype,
                chunks=(min(1000, n_sequences), shape[1], shape[2]),
                compression='gzip',
                compression_opts=4
            )
            
            # Also store the original indices for reference
            f_out.create_dataset('original_indices', data=selected_indices)
            
            # Store metadata
            f_out.attrs['source_file'] = h5_path
            f_out.attrs['n_sequences'] = n_sequences
            f_out.attrs['seed'] = seed
            f_out.attrs['extraction_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Copy sequences in batches for efficiency with tqdm
            batch_size = 5000  # Larger batches for better I/O
            
            with tqdm(total=n_sequences, desc="Extracting sequences", unit="seq") as pbar:
                for i in range(0, n_sequences, batch_size):
                    batch_end = min(i + batch_size, n_sequences)
                    batch_indices = selected_indices[i:batch_end]
                    
                    # Read batch - use list comprehension which is often faster
                    # for non-contiguous reads
                    batch_data = np.stack([input_ids_src[idx] for idx in batch_indices])
                    
                    # Write batch
                    input_ids_dst[i:batch_end] = batch_data
                    pbar.update(batch_end - i)
    
    elapsed = time.time() - start_time
    logger.info(f"Extraction complete in {elapsed:.1f}s ({n_sequences/elapsed:,.0f} seq/s)")
    
    return n_sequences


def write_summary(
    output_path: str,
    h5_path: str,
    criteria: Optional[Set[str]],
    position_criteria: Optional[List[Dict]],
    n_matching: int,
    n_extracted: int,
    indices: np.ndarray
) -> None:
    """Write a summary file with extraction details."""
    summary_path = output_path.replace('.h5', '_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("H5 SEQUENCE EXTRACTION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Source file:      {h5_path}\n")
        f.write(f"Output file:      {output_path}\n")
        
        if position_criteria:
            f.write(f"Mode:             Config-based position criteria\n")
            f.write(f"Criteria count:   {len(position_criteria)}\n")
        else:
            f.write(f"Mode:             Legacy named criteria\n")
            f.write(f"Criteria:         {', '.join(sorted(criteria)) if criteria else 'None'}\n")
        
        f.write(f"Matching found:   {n_matching:,}\n")
        f.write(f"Sequences saved:  {n_extracted:,}\n")
        f.write(f"Extraction time:  {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("CRITERIA DEFINITIONS\n")
        f.write("-" * 40 + "\n")
        
        if position_criteria:
            for c in position_criteria:
                label = c.get('label', f"pos_{c['position']}")
                f.write(f"  {label}: age at position {c['position']} in {c['age_min']}-{c['age_max']}\n")
        else:
            if criteria and 'childhood_start' in criteria:
                f.write("  childhood_start: age at index 6 in 0-9\n")
            if criteria and 'full_length' in criteria:
                f.write("  full_length: token at index 1023 != 0 OR age at index 1023 != 0\n")
            if criteria and 'end_of_life' in criteria:
                f.write("  end_of_life: age at index 1023 in 70-99\n")
            if criteria and 'decade_70' in criteria:
                f.write("  decade_70: age at index 1023 in 70-79\n")
            if criteria and 'decade_80' in criteria:
                f.write("  decade_80: age at index 1023 in 80-89\n")
            if criteria and 'decade_90' in criteria:
                f.write("  decade_90: age at index 1023 in 90-99\n")
        f.write("\n")
        
        # Index statistics
        if len(indices) > 0:
            f.write("INDEX STATISTICS (of matching sequences)\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Min index:      {min(indices):,}\n")
            f.write(f"  Max index:      {max(indices):,}\n")
            f.write(f"  Index range:    {max(indices) - min(indices):,}\n")
    
    logger.info(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract sequences matching criteria from H5 datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Legacy criteria options (comma-separated):
  childhood_start  - age at index 6 is in 0-9
  full_length      - token/age at index 1023 is non-zero
  end_of_life      - age at index 1023 is in 70-99
  decade_70        - age at index 1023 is in 70-79
  decade_80        - age at index 1023 is in 80-89
  decade_90        - age at index 1023 is in 90-99
  all              - shorthand for childhood_start,full_length,end_of_life

Config-based mode (recommended for complex criteria):
  Use --config to specify a JSON file with position-based age criteria.
  Use --generate_config to create a default lifespan config.
  
  Config format:
  {
    "position_age_criteria": [
      {"position": 6, "age_min": 0, "age_max": 9, "label": "childhood"},
      {"position": 100, "age_min": 10, "age_max": 19, "label": "teens"},
      ...
    ]
  }

Examples:
  # Legacy mode
  python h5_sequence_extractor.py --h5_file encoded.h5 --output extracted.h5 \\
      --n_sequences 10000 --criteria childhood_start,full_length,end_of_life
  
  # Generate default lifespan config
  python h5_sequence_extractor.py --generate_config lifespan_criteria.json
  
  # Config-based mode
  python h5_sequence_extractor.py --h5_file encoded.h5 --output extracted.h5 \\
      --n_sequences 10000 --config lifespan_criteria.json
        """
    )
    
    parser.add_argument("--h5_file", 
                        help="Path to source HDF5 file")
    parser.add_argument("--output",
                        help="Path to output HDF5 file")
    parser.add_argument("--n_sequences", type=int, default=10000,
                        help="Number of sequences to extract (default: 10000)")
    parser.add_argument("--criteria", type=str, 
                        default=None,
                        help="Comma-separated legacy criteria to apply")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON/YAML config file with position-based age criteria")
    parser.add_argument("--generate_config", type=str, default=None,
                        help="Generate default lifespan config and save to this path, then exit")
    parser.add_argument("--n_workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="Chunk size for parallel processing (default: 100000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--find_only", action="store_true",
                        help="Only find matching indices, don't extract")
    parser.add_argument("--save_indices", type=str, default=None,
                        help="Save matching indices to a numpy file")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential processing (slower but memory-safe)")
    
    args = parser.parse_args()
    
    # Handle --generate_config
    if args.generate_config:
        config = create_default_lifespan_config()
        save_config(config, args.generate_config)
        logger.info("Generated default lifespan config. You can edit it and use with --config")
        return
    
    # Validate required arguments
    if not args.h5_file:
        logger.error("--h5_file is required (unless using --generate_config)")
        sys.exit(1)
    
    if not args.output and not args.find_only:
        logger.error("--output is required (unless using --find_only)")
        sys.exit(1)
    
    # Validate input file
    if not os.path.exists(args.h5_file):
        logger.error(f"H5 file not found: {args.h5_file}")
        sys.exit(1)
    
    # Determine mode: config-based or legacy
    position_criteria = None
    criteria = None
    
    if args.config:
        # Config-based mode
        logger.info(f"Loading config from: {args.config}")
        config = load_config(args.config)
        position_criteria = config['position_age_criteria']
        logger.info(f"Loaded {len(position_criteria)} position-based criteria")
    else:
        # Legacy mode
        if args.criteria:
            criteria_input = args.criteria.lower().strip()
            if criteria_input == 'all':
                criteria = {'childhood_start', 'full_length', 'end_of_life'}
            else:
                criteria = set(c.strip() for c in criteria_input.split(','))
        else:
            # Default legacy criteria
            criteria = {'childhood_start', 'full_length', 'end_of_life'}
        
        valid_criteria = {
            'childhood_start', 'full_length', 'end_of_life', 
            'decade_70', 'decade_80', 'decade_90', 'all'
        }
        invalid = criteria - valid_criteria
        if invalid:
            logger.error(f"Invalid criteria: {invalid}")
            logger.error(f"Valid options: {valid_criteria}")
            sys.exit(1)
    
    logger.info(f"Source file: {args.h5_file}")
    if args.output:
        logger.info(f"Output file: {args.output}")
    if criteria:
        logger.info(f"Criteria: {criteria}")
    logger.info(f"Target sequences: {args.n_sequences:,}")
    
    # Find matching indices
    # Request a bit more than needed in case of duplicates or issues
    max_indices = args.n_sequences * 2 if not args.find_only else None
    
    matching_indices = find_all_matching_indices(
        h5_path=args.h5_file,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        criteria=criteria,
        position_criteria=position_criteria,
        max_indices=max_indices,
        sequential=args.sequential
    )
    
    n_matching = len(matching_indices)
    logger.info(f"Found {n_matching:,} sequences matching all criteria")
    
    # Save indices if requested
    if args.save_indices:
        np.save(args.save_indices, np.array(matching_indices))
        logger.info(f"Indices saved to: {args.save_indices}")
    
    if args.find_only:
        logger.info("Find-only mode: no extraction performed")
        return
    
    if n_matching == 0:
        logger.error("No matching sequences found! Cannot extract.")
        sys.exit(1)
    
    # Extract sequences
    n_extracted = extract_sequences(
        h5_path=args.h5_file,
        output_path=args.output,
        indices=matching_indices,
        n_sequences=args.n_sequences,
        seed=args.seed
    )
    
    # Write summary
    write_summary(
        output_path=args.output,
        h5_path=args.h5_file,
        criteria=criteria,
        position_criteria=position_criteria,
        n_matching=n_matching,
        n_extracted=n_extracted,
        indices=matching_indices
    )
    
    # Also save the config used (for reproducibility)
    if position_criteria:
        config_copy_path = args.output.replace('.h5', '_criteria.json')
        with open(config_copy_path, 'w') as f:
            json.dump({"position_age_criteria": position_criteria}, f, indent=2)
        logger.info(f"Criteria config saved to: {config_copy_path}")
    
    logger.info("Done!")
    logger.info(f"Extracted {n_extracted:,} sequences to {args.output}")


if __name__ == "__main__":
    main()
