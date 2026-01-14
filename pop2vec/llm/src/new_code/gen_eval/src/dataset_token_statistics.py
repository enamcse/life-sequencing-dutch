#!/usr/bin/env python3
"""
Dataset Token Statistics - Analyze vocabulary and token usage across multiple datasets.

For each dataset (D0, D1, D2, D3, D4, etc.), this script:
1. Loads the vocabulary file (vocab.csv with columns: TOKEN, CATEGORY, ID)
2. Scans sequence files (encoded.h5) to compute:
   - n_people: How many people have this token in their sequence (unique)
   - n_observation: Total occurrences of this token across all sequences

The script handles the folder structure:
    dataset_root/
        vocab.csv (or vocab_v0.csv)
        encoding=mlm/
            masking=random/
                encoded.h5
            masking=event/
                encoded.h5
        encoding=nomlm/
            masking=random/
                encoded.h5

Output:
    - Enhanced vocab CSV with columns like: mlm_random_n_people, mlm_random_n_observation, etc.
    - Metadata JSON with total people and observations per file

Usage:
    python dataset_token_statistics.py --datasets_config datasets.yaml --output_dir ./stats_output
    
    # Or specify datasets directly
    python dataset_token_statistics.py \\
        --D0 /path/to/D0 \\
        --D1 /path/to/D1 \\
        --D3_parent_sibling /path/to/D3_parent_sibling \\
        --output_dir ./stats_output
"""

import argparse
import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import h5py
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# Dataset configurations
@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    root_path: str
    vocab_file: str = "vocab.csv"  # or vocab_v0.csv


@dataclass  
class SequenceFileInfo:
    """Information about a sequence file."""
    path: str
    encoding: str  # 'mlm' or 'nomlm'
    masking: str   # 'random' or 'event'
    prefix: str    # e.g., 'mlm_random', 'nomlm_random'


def find_vocab_file(root_path: str) -> Optional[str]:
    """Find vocabulary file in the dataset root."""
    possible_names = ['vocab.csv', 'vocab_v0.csv', 'vocabulary.csv']
    
    for name in possible_names:
        path = os.path.join(root_path, name)
        if os.path.exists(path):
            return path
    
    return None


def find_sequence_files(root_path: str) -> List[SequenceFileInfo]:
    """Find all encoded.h5 files in the dataset structure."""
    files = []
    
    # Check for encoding folders
    for encoding in ['mlm', 'nomlm']:
        encoding_dir = os.path.join(root_path, f'encoding={encoding}')
        
        if not os.path.exists(encoding_dir):
            continue
        
        # Check for masking folders
        for masking in ['random', 'event']:
            masking_dir = os.path.join(encoding_dir, f'masking={masking}')
            h5_path = os.path.join(masking_dir, 'encoded.h5')
            
            if os.path.exists(h5_path):
                prefix = f'{encoding}_{masking}'
                files.append(SequenceFileInfo(
                    path=h5_path,
                    encoding=encoding,
                    masking=masking,
                    prefix=prefix
                ))
        
        # Also check directly in encoding folder (for datasets without masking subfolder)
        direct_h5 = os.path.join(encoding_dir, 'encoded.h5')
        if os.path.exists(direct_h5):
            # Use 'random' as default masking for nomlm
            prefix = f'{encoding}_random'
            files.append(SequenceFileInfo(
                path=direct_h5,
                encoding=encoding,
                masking='random',
                prefix=prefix
            ))
    
    # Also check for h5 file directly in root
    root_h5 = os.path.join(root_path, 'encoded.h5')
    if os.path.exists(root_h5):
        files.append(SequenceFileInfo(
            path=root_h5,
            encoding='unknown',
            masking='unknown',
            prefix='default'
        ))
    
    return files


def process_chunk_for_token_stats(
    h5_path: str,
    start_idx: int,
    end_idx: int,
    pad_id: int = 0
) -> Tuple[Counter, Counter]:
    """
    Process a chunk of sequences and count token occurrences.
    
    Returns:
        Tuple of (n_people_counter, n_observation_counter)
        - n_people_counter: token_id -> number of sequences containing this token
        - n_observation_counter: token_id -> total occurrences
    """
    n_people_counter = Counter()
    n_observation_counter = Counter()
    
    try:
        with h5py.File(h5_path, 'r') as f:
            input_ids = f['input_ids']
            
            # Handle different data shapes
            # Could be (N, 4, seq_len) or (N, seq_len)
            if len(input_ids.shape) == 3:
                # Shape: (N, 4, seq_len) - tokens are at index 0
                tokens = input_ids[start_idx:end_idx, 0, :]
            else:
                # Shape: (N, seq_len)
                tokens = input_ids[start_idx:end_idx, :]
            
            for seq_idx in range(tokens.shape[0]):
                seq_tokens = tokens[seq_idx, :]
                
                # Exclude PAD tokens
                valid_tokens = seq_tokens[seq_tokens != pad_id]
                
                # Count unique tokens for n_people (each person counted once per token type)
                unique_tokens = set(valid_tokens.tolist())
                for token_id in unique_tokens:
                    n_people_counter[token_id] += 1
                
                # Count all occurrences for n_observation
                for token_id in valid_tokens:
                    n_observation_counter[int(token_id)] += 1
    
    except Exception as e:
        logger.error(f"Error processing chunk [{start_idx}:{end_idx}] in {h5_path}: {e}")
    
    return n_people_counter, n_observation_counter


def compute_token_statistics(
    h5_path: str,
    n_workers: int = 8,
    chunk_size: int = 50000,
    pad_id: int = 0
) -> Tuple[Counter, Counter, int, int]:
    """
    Compute token statistics for a sequence file.
    
    Returns:
        Tuple of (n_people_counter, n_observation_counter, total_people, total_observations)
    """
    logger.info(f"Processing: {h5_path}")
    
    # Get dataset info
    with h5py.File(h5_path, 'r') as f:
        if 'input_ids' not in f:
            logger.warning(f"No 'input_ids' key in {h5_path}")
            return Counter(), Counter(), 0, 0
        
        input_ids = f['input_ids']
        n_sequences = input_ids.shape[0]
        
        if len(input_ids.shape) == 3:
            seq_len = input_ids.shape[2]
        else:
            seq_len = input_ids.shape[1]
        
        logger.info(f"  Shape: {input_ids.shape}, sequences: {n_sequences:,}, seq_len: {seq_len}")
    
    # Create chunks
    chunks = []
    for start in range(0, n_sequences, chunk_size):
        end = min(start + chunk_size, n_sequences)
        chunks.append((start, end))
    
    # Process chunks in parallel
    all_n_people = Counter()
    all_n_observation = Counter()
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_chunk_for_token_stats, h5_path, start, end, pad_id): (start, end)
            for start, end in chunks
        }
        
        with tqdm(total=len(chunks), desc=f"  Chunks", leave=False) as pbar:
            for future in as_completed(futures):
                n_people_chunk, n_obs_chunk = future.result()
                all_n_people.update(n_people_chunk)
                all_n_observation.update(n_obs_chunk)
                pbar.update(1)
    
    elapsed = time.time() - start_time
    total_people = n_sequences
    total_observations = sum(all_n_observation.values())
    
    logger.info(f"  Complete in {elapsed:.1f}s")
    logger.info(f"  Total people: {total_people:,}")
    logger.info(f"  Total observations: {total_observations:,}")
    logger.info(f"  Unique tokens: {len(all_n_observation):,}")
    
    return all_n_people, all_n_observation, total_people, total_observations


def process_dataset(
    dataset: DatasetConfig,
    output_dir: str,
    n_workers: int = 8,
    chunk_size: int = 50000,
    pad_id: int = 0
) -> Optional[str]:
    """
    Process a single dataset and create enhanced vocabulary file.
    
    Returns path to output vocab file, or None if failed.
    """
    logger.info("="*70)
    logger.info(f"Processing Dataset: {dataset.name}")
    logger.info(f"Root: {dataset.root_path}")
    logger.info("="*70)
    
    if not os.path.exists(dataset.root_path):
        logger.error(f"Dataset root not found: {dataset.root_path}")
        return None
    
    # Find vocabulary file
    vocab_path = find_vocab_file(dataset.root_path)
    if vocab_path is None:
        logger.error(f"No vocabulary file found in {dataset.root_path}")
        return None
    
    logger.info(f"Vocabulary file: {vocab_path}")
    
    # Load vocabulary
    vocab_df = pd.read_csv(vocab_path)
    logger.info(f"Vocabulary size: {len(vocab_df)}")
    
    # Find sequence files
    seq_files = find_sequence_files(dataset.root_path)
    
    if not seq_files:
        logger.warning(f"No sequence files found in {dataset.root_path}")
        # Just copy the vocab file as-is
        output_path = os.path.join(output_dir, f'{dataset.name}_vocab_stats.csv')
        vocab_df.to_csv(output_path, index=False)
        return output_path
    
    logger.info(f"Found {len(seq_files)} sequence file(s):")
    for sf in seq_files:
        logger.info(f"  - {sf.prefix}: {sf.path}")
    
    # Process each sequence file
    metadata = {
        'dataset_name': dataset.name,
        'root_path': dataset.root_path,
        'vocab_path': vocab_path,
        'vocab_size': len(vocab_df),
        'sequence_files': {}
    }
    
    for seq_file in tqdm(seq_files, desc=f"Processing {dataset.name} files"):
        n_people_counter, n_obs_counter, total_people, total_obs = compute_token_statistics(
            seq_file.path,
            n_workers=n_workers,
            chunk_size=chunk_size,
            pad_id=pad_id
        )
        
        # Add columns to vocab dataframe
        col_n_people = f'{seq_file.prefix}_n_people'
        col_n_obs = f'{seq_file.prefix}_n_observation'
        
        vocab_df[col_n_people] = vocab_df['ID'].map(lambda x: n_people_counter.get(x, 0))
        vocab_df[col_n_obs] = vocab_df['ID'].map(lambda x: n_obs_counter.get(x, 0))
        
        # Store metadata
        metadata['sequence_files'][seq_file.prefix] = {
            'path': seq_file.path,
            'encoding': seq_file.encoding,
            'masking': seq_file.masking,
            'total_people': total_people,
            'total_observations': total_obs,
            'unique_tokens_used': len(n_obs_counter),
        }
    
    # Save enhanced vocab
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{dataset.name}_vocab_stats.csv')
    vocab_df.to_csv(output_path, index=False)
    logger.info(f"Saved enhanced vocab: {output_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f'{dataset.name}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {metadata_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute token statistics across multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using config file
    python dataset_token_statistics.py --config datasets.yaml --output_dir ./stats_output
    
    # Specifying datasets directly
    python dataset_token_statistics.py \\
        --D0 /path/to/D0 \\
        --D3_full_pop /path/to/D3_full_pop \\
        --output_dir ./stats_output

Config file format (YAML):
    datasets:
      D0: /path/to/D0
      D1: /path/to/D1
      D2: /path/to/D2
      D3_parent_sibling: /path/to/D3_parent_sibling
      D3_full_pop: /path/to/D3_full_pop
      D4: /path/to/D4
      D4_bd: /path/to/D4_bd
        """
    )
    
    # Config file option
    parser.add_argument("--config", help="Path to YAML config file with dataset paths")
    
    # Direct dataset path options
    parser.add_argument("--D0", help="Path to D0 dataset root")
    parser.add_argument("--D1", help="Path to D1 dataset root")
    parser.add_argument("--D2", help="Path to D2 dataset root")
    parser.add_argument("--D3_parent_sibling", help="Path to D3_parent_sibling dataset root")
    parser.add_argument("--D3_full_pop", help="Path to D3_full_pop dataset root")
    parser.add_argument("--D4", help="Path to D4 dataset root")
    parser.add_argument("--D4_bd", help="Path to D4_bd dataset root")
    
    # General options
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--n_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--chunk_size", type=int, default=50000, help="Chunk size for processing")
    parser.add_argument("--pad_id", type=int, default=0, help="PAD token ID")
    
    args = parser.parse_args()
    
    # Build list of datasets
    datasets = []
    
    # From config file
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        for name, path in config.get('datasets', {}).items():
            if path and path.strip():
                datasets.append(DatasetConfig(name=name, root_path=path))
    
    # From command line arguments
    dataset_args = ['D0', 'D1', 'D2', 'D3_parent_sibling', 'D3_full_pop', 'D4', 'D4_bd']
    for name in dataset_args:
        path = getattr(args, name, None)
        if path and path.strip():
            # Check if already added from config
            if not any(d.name == name for d in datasets):
                datasets.append(DatasetConfig(name=name, root_path=path))
    
    if not datasets:
        logger.error("No datasets specified. Use --config or --D0, --D1, etc.")
        return
    
    logger.info(f"Processing {len(datasets)} dataset(s)")
    for ds in datasets:
        logger.info(f"  - {ds.name}: {ds.root_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dataset
    results = []
    for dataset in tqdm(datasets, desc="Datasets"):
        output_path = process_dataset(
            dataset,
            args.output_dir,
            n_workers=args.n_workers,
            chunk_size=args.chunk_size,
            pad_id=args.pad_id
        )
        if output_path:
            results.append((dataset.name, output_path))
    
    # Create summary
    logger.info("\n" + "="*70)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*70)
    
    for name, path in results:
        logger.info(f"  {name}: {path}")
    
    # Create combined summary
    summary_path = os.path.join(args.output_dir, 'all_datasets_summary.json')
    summary = {
        'datasets_processed': len(results),
        'datasets': {name: path for name, path in results},
        'output_dir': args.output_dir,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved: {summary_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
