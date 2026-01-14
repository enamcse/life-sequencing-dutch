#!/usr/bin/env python3
"""
Create Generative Datasets (GD0-GD4 and GDB0-GDB4)

Creates age-stratified datasets for generative evaluation:
- GD0: Childhood/Young (age 1-30 at position 1000)
- GD1: Middle-age (age 30-49 at position 1000)
- GD2: Late middle-age (age 50-69 at position 1000)
- GD3: Old age (age 70-99 at position 1000)
- GD4: Mixed sampling (20% 0-29, 25% 30-49, 25% 50-69, 20% 70-99, 10% with death)

GDB0-GDB4 are the corresponding birthday-token versions (same row indices, different H5 file).

Usage:
    python create_generative_datasets.py --config generative_datasets_config.yaml
    python create_generative_datasets.py --config generative_datasets_config.yaml --datasets GD0 GD1
    python create_generative_datasets.py --config generative_datasets_config.yaml --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

import h5py
import numpy as np
import yaml
from tqdm import tqdm

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# =============================================================================
# Token/Vocab Utilities
# =============================================================================

def load_vocab(vocab_path: str) -> Dict[str, int]:
    """Load vocabulary and return token -> id mapping."""
    import pandas as pd
    df = pd.read_csv(vocab_path)
    return dict(zip(df['token'], df['token_id']))


def get_special_token_ids(vocab_path: str, tokens: List[str]) -> Dict[str, int]:
    """Get token IDs for special tokens."""
    vocab = load_vocab(vocab_path)
    result = {}
    for token in tokens:
        if token in vocab:
            result[token] = vocab[token]
        else:
            logger.warning(f"Token '{token}' not found in vocabulary")
            result[token] = -1
    return result


# =============================================================================
# Criteria Checking - Vectorized
# =============================================================================

def find_matching_indices_chunk(
    h5_path: str,
    start_idx: int,
    end_idx: int,
    position_criteria: List[Dict],
    pad_id: int = 0
) -> np.ndarray:
    """
    Find indices matching position-based age criteria (vectorized).
    
    Args:
        h5_path: Path to HDF5 file
        start_idx: Start index (inclusive)
        end_idx: End index (exclusive)
        position_criteria: List of {position, age_min, age_max}
        pad_id: PAD token ID
    
    Returns:
        Array of global indices matching ALL criteria
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            input_ids = f['input_ids']
            chunk_size = end_idx - start_idx
            
            # Get required positions
            positions = [c['position'] for c in position_criteria]
            max_pos = max(positions)
            
            # Check sequence length
            seq_len = input_ids.shape[2]
            if max_pos >= seq_len:
                logger.warning(f"Position {max_pos} exceeds sequence length {seq_len}")
                return np.array([], dtype=np.int64)
            
            # Read tokens (stream 0) and ages (stream 2) at required positions
            # Shape: (chunk_size, n_positions)
            tokens = input_ids[start_idx:end_idx, 0, positions]
            ages = input_ids[start_idx:end_idx, 2, positions]
            
            # Start with all True mask
            valid_mask = np.ones(chunk_size, dtype=bool)
            
            for i, criterion in enumerate(position_criteria):
                pos_ages = ages[:, i]
                pos_tokens = tokens[:, i]
                age_min = criterion['age_min']
                age_max = criterion['age_max']
                
                # Apply age range filter
                # Special case: if age_min > 0, we also need to exclude PAD tokens
                # (where both age=0 AND token=PAD)
                if age_min > 0:
                    # Simple: age must be in range
                    valid_mask &= (pos_ages >= age_min) & (pos_ages <= age_max)
                else:
                    # For age_min=0, exclude only if both age=0 AND token=PAD
                    is_pad = (pos_ages == 0) & (pos_tokens == pad_id)
                    in_range = (pos_ages >= age_min) & (pos_ages <= age_max)
                    valid_mask &= in_range & ~is_pad
            
            # Return global indices
            matching_local = np.where(valid_mask)[0]
            return matching_local + start_idx
    
    except Exception as e:
        logger.error(f"Error processing chunk [{start_idx}:{end_idx}]: {e}")
        return np.array([], dtype=np.int64)


def find_death_sequences_chunk(
    h5_path: str,
    start_idx: int,
    end_idx: int,
    death_token_id: int
) -> np.ndarray:
    """
    Find indices of sequences containing the death token.
    
    Args:
        h5_path: Path to HDF5 file
        start_idx: Start index
        end_idx: End index
        death_token_id: ID of the death token
    
    Returns:
        Array of global indices containing death token
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            tokens = f['input_ids'][start_idx:end_idx, 0, :]
            # Check if death token exists in any position
            has_death = np.any(tokens == death_token_id, axis=1)
            matching_local = np.where(has_death)[0]
            return matching_local + start_idx
    
    except Exception as e:
        logger.error(f"Error processing chunk [{start_idx}:{end_idx}]: {e}")
        return np.array([], dtype=np.int64)


def find_all_matching_indices(
    h5_path: str,
    position_criteria: List[Dict],
    n_workers: int = 8,
    chunk_size: int = 100000,
    pad_id: int = 0
) -> np.ndarray:
    """Find all indices matching criteria using parallel processing."""
    with h5py.File(h5_path, 'r') as f:
        n_sequences = f['input_ids'].shape[0]
    
    # Create chunks
    chunks = []
    for start in range(0, n_sequences, chunk_size):
        end = min(start + chunk_size, n_sequences)
        chunks.append((start, end))
    
    logger.info(f"Processing {len(chunks)} chunks of ~{chunk_size:,} sequences each")
    
    all_indices = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                find_matching_indices_chunk, h5_path, start, end, position_criteria, pad_id
            ): (start, end)
            for start, end in chunks
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Finding matches"):
            indices = future.result()
            if len(indices) > 0:
                all_indices.append(indices)
    
    if all_indices:
        result = np.concatenate(all_indices)
        result.sort()
        return result
    return np.array([], dtype=np.int64)


def find_death_sequences(
    h5_path: str,
    death_token_id: int,
    n_workers: int = 8,
    chunk_size: int = 100000
) -> np.ndarray:
    """Find all sequences containing death token."""
    with h5py.File(h5_path, 'r') as f:
        n_sequences = f['input_ids'].shape[0]
    
    chunks = [(start, min(start + chunk_size, n_sequences)) 
              for start in range(0, n_sequences, chunk_size)]
    
    all_indices = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(find_death_sequences_chunk, h5_path, start, end, death_token_id): (start, end)
            for start, end in chunks
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Finding death sequences"):
            indices = future.result()
            if len(indices) > 0:
                all_indices.append(indices)
    
    if all_indices:
        result = np.concatenate(all_indices)
        result.sort()
        return result
    return np.array([], dtype=np.int64)


# =============================================================================
# Extraction Functions
# =============================================================================

def extract_to_h5(
    source_h5: str,
    output_h5: str,
    indices: np.ndarray,
    batch_size: int = 10000
) -> int:
    """
    Extract sequences at given indices to a new H5 file.
    
    Args:
        source_h5: Source H5 file path
        output_h5: Output H5 file path
        indices: Array of indices to extract
        batch_size: Batch size for reading/writing
    
    Returns:
        Number of sequences extracted
    """
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    
    indices = np.sort(indices)
    n_sequences = len(indices)
    
    logger.info(f"Extracting {n_sequences:,} sequences to {output_h5}")
    
    with h5py.File(source_h5, 'r') as f_in:
        input_ids = f_in['input_ids']
        seq_len = input_ids.shape[2]
        
        # Check for sequence_id dataset
        has_sequence_id = 'sequence_id' in f_in
        
        with h5py.File(output_h5, 'w') as f_out:
            # Create output datasets
            out_input_ids = f_out.create_dataset(
                'input_ids',
                shape=(n_sequences, 4, seq_len),
                dtype=input_ids.dtype,
                chunks=(min(1000, n_sequences), 4, seq_len)
            )
            
            if has_sequence_id:
                seq_ids_in = f_in['sequence_id']
                out_seq_ids = f_out.create_dataset(
                    'sequence_id',
                    shape=(n_sequences,),
                    dtype=seq_ids_in.dtype
                )
            
            # Store original indices for reference
            f_out.create_dataset('original_indices', data=indices)
            
            # Extract in batches
            for batch_start in tqdm(range(0, n_sequences, batch_size), desc="Extracting"):
                batch_end = min(batch_start + batch_size, n_sequences)
                batch_indices = indices[batch_start:batch_end]
                
                # Read source data (sorted indices for efficiency)
                data = input_ids[batch_indices]
                out_input_ids[batch_start:batch_end] = data
                
                if has_sequence_id:
                    seq_ids = seq_ids_in[batch_indices]
                    out_seq_ids[batch_start:batch_end] = seq_ids
    
    logger.info(f"Extraction complete: {n_sequences:,} sequences")
    return n_sequences


def extract_paired_h5(
    source_primary: str,
    source_secondary: str,
    output_primary: str,
    output_secondary: str,
    indices: np.ndarray,
    batch_size: int = 10000
) -> Tuple[int, int]:
    """
    Extract from paired H5 files (primary + birthday version).
    Uses indices from primary file for both extractions.
    
    Returns:
        (n_primary, n_secondary) extracted
    """
    n_primary = extract_to_h5(source_primary, output_primary, indices, batch_size)
    n_secondary = extract_to_h5(source_secondary, output_secondary, indices, batch_size)
    return n_primary, n_secondary


# =============================================================================
# Dataset Creation
# =============================================================================

def create_simple_dataset(
    config: Dict,
    dataset_name: str,
    dataset_config: Dict,
    output_dir: str,
    n_sequences: int,
    n_workers: int,
    pad_id: int,
    seed: int,
    source_h5: str,
    source_h5_birthday: Optional[str] = None,
    dry_run: bool = False
) -> Dict:
    """
    Create a simple age-stratified dataset (GD0-GD3).
    
    Returns:
        Summary dict with stats
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Creating dataset: {dataset_name}")
    logger.info(f"Description: {dataset_config.get('description', 'N/A')}")
    logger.info(f"{'='*60}")
    
    position_criteria = dataset_config['position_age_criteria']
    logger.info(f"Position criteria: {position_criteria}")
    
    # Find matching indices
    start_time = time.time()
    matching_indices = find_all_matching_indices(
        source_h5, position_criteria, n_workers, pad_id=pad_id
    )
    search_time = time.time() - start_time
    
    logger.info(f"Found {len(matching_indices):,} matching sequences in {search_time:.1f}s")
    
    if len(matching_indices) < n_sequences:
        logger.warning(f"Only {len(matching_indices)} sequences match, requested {n_sequences}")
        n_sequences = len(matching_indices)
    
    if dry_run:
        return {
            'dataset': dataset_name,
            'n_matching': len(matching_indices),
            'n_to_extract': n_sequences,
            'dry_run': True
        }
    
    # Random sample
    np.random.seed(seed)
    selected_indices = np.random.choice(matching_indices, size=n_sequences, replace=False)
    selected_indices = np.sort(selected_indices)
    
    # Create output paths
    output_primary = os.path.join(output_dir, dataset_name, f"{dataset_name}.h5")
    
    # Extract primary
    start_time = time.time()
    n_extracted = extract_to_h5(source_h5, output_primary, selected_indices)
    extract_time = time.time() - start_time
    
    # Extract birthday version if available
    n_birthday = 0
    if source_h5_birthday:
        output_birthday = os.path.join(output_dir, f"{dataset_name}B", f"{dataset_name}B.h5")
        n_birthday = extract_to_h5(source_h5_birthday, output_birthday, selected_indices)
    
    # Save indices for reproducibility
    indices_path = os.path.join(output_dir, dataset_name, f"{dataset_name}_indices.npy")
    np.save(indices_path, selected_indices)
    
    # Save metadata
    metadata = {
        'dataset_name': dataset_name,
        'description': dataset_config.get('description', ''),
        'position_criteria': position_criteria,
        'n_matching': len(matching_indices),
        'n_extracted': n_extracted,
        'seed': seed,
        'source_h5': source_h5,
        'has_birthday_version': source_h5_birthday is not None,
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    metadata_path = os.path.join(output_dir, dataset_name, f"{dataset_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Dataset {dataset_name} created:")
    logger.info(f"  Primary: {output_primary} ({n_extracted:,} sequences)")
    if source_h5_birthday:
        logger.info(f"  Birthday: {output_birthday} ({n_birthday:,} sequences)")
    logger.info(f"  Extraction time: {extract_time:.1f}s")
    
    return {
        'dataset': dataset_name,
        'n_matching': len(matching_indices),
        'n_extracted': n_extracted,
        'n_birthday': n_birthday,
        'primary_path': output_primary,
        'indices_path': indices_path,
        'extract_time': extract_time
    }


def create_mixed_dataset(
    config: Dict,
    dataset_config: Dict,
    output_dir: str,
    n_sequences: int,
    n_workers: int,
    pad_id: int,
    death_token_id: int,
    seed: int,
    source_h5: str,
    source_h5_birthday: Optional[str] = None,
    dry_run: bool = False
) -> Dict:
    """
    Create the mixed dataset (GD4) with proportional sampling.
    
    Proportions:
    - 20% age 1-29 at position 1000
    - 25% age 30-49 at position 1000
    - 25% age 50-69 at position 1000
    - 20% age 70-99 at position 1000
    - 10% sequences with death token
    """
    logger.info(f"\n{'='*60}")
    logger.info("Creating mixed dataset: GD4")
    logger.info(f"Description: {dataset_config.get('description', 'N/A')}")
    logger.info(f"{'='*60}")
    
    proportions = dataset_config['sampling_proportions']
    
    # Define age groups
    age_groups = {
        'age_0_29': {'position': 1000, 'age_min': 1, 'age_max': 29},
        'age_30_49': {'position': 1000, 'age_min': 30, 'age_max': 49},
        'age_50_69': {'position': 1000, 'age_min': 50, 'age_max': 69},
        'age_70_99': {'position': 1000, 'age_min': 70, 'age_max': 99},
    }
    
    # Find indices for each group
    group_indices = {}
    
    for group_name, age_range in age_groups.items():
        criteria = [{'position': 6, 'age_min': 0, 'age_max': 120}, age_range]
        indices = find_all_matching_indices(source_h5, criteria, n_workers, pad_id=pad_id)
        group_indices[group_name] = indices
        logger.info(f"  {group_name}: {len(indices):,} matching sequences")
    
    # Find death sequences
    death_indices = find_death_sequences(source_h5, death_token_id, n_workers)
    group_indices['with_death'] = death_indices
    logger.info(f"  with_death: {len(death_indices):,} matching sequences")
    
    # Calculate samples per group
    samples_per_group = {}
    for group_name, prop in proportions.items():
        n_needed = int(n_sequences * prop)
        n_available = len(group_indices.get(group_name, []))
        samples_per_group[group_name] = min(n_needed, n_available)
    
    logger.info(f"\nSampling plan:")
    for group, n in samples_per_group.items():
        logger.info(f"  {group}: {n} samples ({proportions[group]*100:.0f}%)")
    
    if dry_run:
        return {
            'dataset': 'GD4',
            'group_matches': {k: len(v) for k, v in group_indices.items()},
            'samples_per_group': samples_per_group,
            'dry_run': True
        }
    
    # Sample from each group
    np.random.seed(seed)
    all_selected = []
    
    for group_name, n_samples in samples_per_group.items():
        if n_samples > 0 and len(group_indices[group_name]) > 0:
            selected = np.random.choice(group_indices[group_name], size=n_samples, replace=False)
            all_selected.append(selected)
    
    # Combine and deduplicate
    selected_indices = np.unique(np.concatenate(all_selected))
    selected_indices = np.sort(selected_indices)
    
    logger.info(f"\nTotal unique indices selected: {len(selected_indices):,}")
    
    # Create output
    output_primary = os.path.join(output_dir, "GD4", "GD4.h5")
    n_extracted = extract_to_h5(source_h5, output_primary, selected_indices)
    
    # Extract birthday version if available
    n_birthday = 0
    if source_h5_birthday:
        output_birthday = os.path.join(output_dir, "GD4B", "GD4B.h5")
        n_birthday = extract_to_h5(source_h5_birthday, output_birthday, selected_indices)
    
    # Save indices
    indices_path = os.path.join(output_dir, "GD4", "GD4_indices.npy")
    np.save(indices_path, selected_indices)
    
    # Save metadata
    metadata = {
        'dataset_name': 'GD4',
        'description': dataset_config.get('description', ''),
        'proportions': proportions,
        'samples_per_group': samples_per_group,
        'group_matches': {k: len(v) for k, v in group_indices.items()},
        'n_extracted': n_extracted,
        'seed': seed,
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    metadata_path = os.path.join(output_dir, "GD4", "GD4_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        'dataset': 'GD4',
        'n_extracted': n_extracted,
        'n_birthday': n_birthday,
        'samples_per_group': samples_per_group,
        'primary_path': output_primary
    }


# =============================================================================
# Main
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Create generative evaluation datasets (GD0-GD4, GDB0-GDB4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create all datasets
    python create_generative_datasets.py --config generative_datasets_config.yaml
    
    # Create specific datasets
    python create_generative_datasets.py --config config.yaml --datasets GD0 GD1
    
    # Dry run (show what would be created)
    python create_generative_datasets.py --config config.yaml --dry-run
        """
    )
    
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to create (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be created without extracting")
    parser.add_argument("--n-sequences", type=int, help="Override number of sequences to extract")
    parser.add_argument("--n-workers", type=int, help="Override number of parallel workers")
    parser.add_argument("--output-dir", help="Override output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get parameters
    source_h5 = config['source_h5']['primary']
    source_h5_birthday = config['source_h5'].get('birthday')
    output_dir = args.output_dir or config['output_dir']
    n_sequences = args.n_sequences or config['extraction']['n_sequences']
    n_workers = args.n_workers or config['extraction']['n_workers']
    seed = config['extraction']['seed']
    vocab_path = config['vocab_path']
    
    # Get special token IDs
    special_tokens = get_special_token_ids(vocab_path, ['[PAD]', config['datasets']['GD4']['death_token']])
    pad_id = special_tokens['[PAD]']
    death_token_id = special_tokens[config['datasets']['GD4']['death_token']]
    
    logger.info(f"PAD token ID: {pad_id}")
    logger.info(f"DEATH token ID: {death_token_id}")
    
    # Determine which datasets to create
    datasets_to_create = args.datasets or list(config['datasets'].keys())
    
    logger.info(f"\nWill create {len(datasets_to_create)} datasets: {datasets_to_create}")
    logger.info(f"Source H5: {source_h5}")
    if source_h5_birthday:
        logger.info(f"Birthday H5: {source_h5_birthday}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Sequences per dataset: {n_sequences:,}")
    
    # Create each dataset
    results = []
    
    for dataset_name in datasets_to_create:
        dataset_config = config['datasets'][dataset_name]
        
        if dataset_config.get('is_mixed', False):
            # Mixed dataset (GD4)
            result = create_mixed_dataset(
                config, dataset_config, output_dir, n_sequences,
                n_workers, pad_id, death_token_id, seed,
                source_h5, source_h5_birthday, args.dry_run
            )
        else:
            # Simple age-stratified dataset (GD0-GD3)
            result = create_simple_dataset(
                config, dataset_name, dataset_config, output_dir, n_sequences,
                n_workers, pad_id, seed, source_h5, source_h5_birthday, args.dry_run
            )
        
        results.append(result)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    for result in results:
        logger.info(f"\n{result['dataset']}:")
        for key, value in result.items():
            if key != 'dataset':
                logger.info(f"  {key}: {value}")
    
    logger.info(f"\n{'='*60}")
    logger.info("Dataset creation complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
