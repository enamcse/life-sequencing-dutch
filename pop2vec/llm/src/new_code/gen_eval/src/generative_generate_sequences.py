#!/usr/bin/env python3
"""
Generate Sequences (GPU Phase)

Generates token sequences from prefixes and saves them to Parquet files.
This script is GPU-intensive and saves raw data for later statistical analysis.

Outputs:
    1. original_sequences.parquet - All original sequences (n persons + n buddies)
       Columns: idx, rinpersoon_id, original_sequence
    2. generated_sequences.parquet - Generated sequences with metadata
       Columns: person_idx, rinpersoon_id, buddy_rinpersoon_id, prefix_len, generation_idx, 
                generated_tokens, original_len, generated_len

Usage:
    python generate_sequences.py --config run_config.yaml
"""

import argparse
import json
import logging
import numpy as np
import os
import time
import torch
import yaml
import h5py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from torch.nn.functional import softmax

# Project imports
from pop2vec.llm.src.new_code.utils import load_special_ids, load_vocab_df
from pop2vec.llm.src.transformer.models import TransformerEncoder

# Logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for sequence generation."""
    # Paths
    model_name: str
    checkpoint_path: str
    data_path: str
    vocab_path: str
    output_dir: str
    sequences_path: str
    
    # Evaluation parameters
    num_people: int = 10
    num_generations: int = 100
    horizon: int = 20
    prefix_lengths: List[int] = None
    
    # Padding exclusion mode: 'none' (use full sequence), 'exclude' (truncate at PAD)
    exclude_padding: bool = True
    
    # Batch size for generation (higher = better GPU utilization, but more memory)
    generation_batch_size: int = 64
    
    # Sampling
    top_k: int = 20
    temperature: float = 1.0
    
    # Special tokens
    pad_token: str = "[PAD]"
    cls_token: str = "[CLS]"
    death_token: str = "[death]_[death]"
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        if self.prefix_lengths is None:
            self.prefix_lengths = [1, 101, 201, 301, 401, 501, 601, 701, 801, 901, 1001]


class SequenceGenerator:
    """GPU-based sequence generator using direct h5 access."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Device: {self.device}")
        
        # Set seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Load vocabulary
        logger.info(f"Loading vocabulary: {config.vocab_path}")
        self.vocab_df = load_vocab_df(config.vocab_path)
        self.vocab_size = len(self.vocab_df)
        logger.info(f"Vocabulary size: {self.vocab_size}")
        
        # Load special tokens
        specials = load_special_ids(
            config.vocab_path,
            pad_token=config.pad_token,
            cls_token=config.cls_token,
            death_token=config.death_token,
        )
        self.pad_id = specials["pad_id"]
        self.cls_id = specials["cls_id"]
        self.death_id = specials["death_id"]
        logger.info(f"Special tokens: PAD={self.pad_id}, CLS={self.cls_id}, DEATH={self.death_id}")
        logger.info(f"Padding exclusion: {config.exclude_padding}")
        
        # Load model
        logger.info(f"Loading model: {config.checkpoint_path}")
        self.model = TransformerEncoder.load_from_checkpoint(
            config.checkpoint_path,
            strict=False
        )
        self.model.eval().to(self.device)
        
        # Open h5 file directly - only use 'input_ids' and 'sequence_id' keys
        logger.info(f"Opening h5 file: {config.data_path}")
        self.h5_file = h5py.File(config.data_path, 'r', libver='latest', swmr=True)
        self.dataset_size = self.h5_file['input_ids'].shape[0]
        self.max_seq_len = self.h5_file['input_ids'].shape[2]  # shape: (N, 4, L)
        logger.info(f"Dataset size: {self.dataset_size}, max_seq_len: {self.max_seq_len}")
    
    def __del__(self):
        """Close h5 file on cleanup."""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
    
    def _find_real_length(self, tokens: np.ndarray) -> int:
        """
        Find real sequence length by looking for first PAD token.
        
        Args:
            tokens: 1D array of token IDs (stream 0 from input_ids)
        
        Returns:
            Length of real tokens (before padding starts)
        """
        if not self.config.exclude_padding:
            return len(tokens)
        
        pad_positions = np.where(tokens == self.pad_id)[0]
        if len(pad_positions) > 0:
            return int(pad_positions[0])
        return len(tokens)
    
    def _load_person_data(self, idx: int) -> Dict:
        """
        Load a person's data directly from h5 file.
        Only uses 'input_ids' and 'sequence_id' keys.
        """
        # input_ids shape: (4, L) for this person
        input_ids = self.h5_file['input_ids'][idx]  # (4, L)
        sequence_id = self.h5_file['sequence_id'][idx]  # rinpersoon_id
        
        # Convert to tensors
        x4 = torch.as_tensor(input_ids, dtype=torch.long)  # (4, L)
        
        # Find real length from token stream (stream 0)
        L_real = self._find_real_length(input_ids[0])
        
        # Create padding mask from real length (1=real, 0=pad)
        pm = torch.ones(L_real, dtype=torch.long)
        
        # Extract the full age stream (stream 2) for position-dependent age lookups
        # The age at position p tells us the age when that token occurred
        age_stream = input_ids[2].tolist()  # Full age stream for all positions
        
        return {
            'idx': int(idx),
            'x4': x4[:, :L_real],
            'pm': pm,
            'L_real': L_real,
            'rinpersoon_id': int(sequence_id),
            'full_sequence': input_ids[0].tolist(),  # Store full original sequence for saving
            'age_stream': age_stream,  # Full age stream for position-dependent age lookups
        }
    
    def _select_people(self) -> Tuple[np.ndarray, np.ndarray]:
        """Select n people and their random buddies."""
        n = self.config.num_people
        total = self.dataset_size
        
        # Select main people
        selected = np.random.choice(total, size=min(n, total), replace=False)
        
        # Select random buddies (different from selected)
        buddies = np.random.choice(total, size=min(n, total), replace=False)
        for i in range(len(buddies)):
            while buddies[i] == selected[i]:
                buddies[i] = np.random.randint(0, total)
        
        return selected, buddies
    
    @torch.no_grad()
    def _generate_tokens_batch(
        self,
        prefixes_4stream: List[torch.Tensor],
        pad_masks: List[torch.Tensor],
        horizon: int,
        batch_size: int = 32,
    ) -> List[List[int]]:
        """
        Generate tokens for multiple sequences in batches.
        
        Args:
            prefixes_4stream: List of prefix tensors, each (4, prefix_len)
            pad_masks: List of padding masks, each (prefix_len,)
            horizon: Number of tokens to generate
            batch_size: Number of sequences to process in parallel
        
        Returns:
            List of generated token sequences
        """
        all_results = []
        n_sequences = len(prefixes_4stream)
        
        for batch_start in range(0, n_sequences, batch_size):
            batch_end = min(batch_start + batch_size, n_sequences)
            batch_prefixes = prefixes_4stream[batch_start:batch_end]
            batch_masks = pad_masks[batch_start:batch_end]
            
            # Pad sequences to same length for batching
            max_prefix_len = max(p.size(1) for p in batch_prefixes)
            
            batch_x = []
            batch_pm = []
            
            for prefix, mask in zip(batch_prefixes, batch_masks):
                curr_len = prefix.size(1)
                if curr_len < max_prefix_len:
                    # Pad on the left with PAD tokens
                    pad_len = max_prefix_len - curr_len
                    pad_tokens = torch.full((4, pad_len), self.pad_id, dtype=torch.long)
                    padded_prefix = torch.cat([pad_tokens, prefix], dim=1)
                    padded_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), mask])
                else:
                    padded_prefix = prefix
                    padded_mask = mask
                
                batch_x.append(padded_prefix)
                batch_pm.append(padded_mask)
            
            # Stack into batches: (B, 4, L) and (B, L)
            x = torch.stack(batch_x).to(self.device)
            pm = torch.stack(batch_pm).to(self.device)
            
            B = x.size(0)
            
            # Track which sequences are still active (haven't hit DEATH)
            active = torch.ones(B, dtype=torch.bool, device=self.device)
            out_tokens = [[] for _ in range(B)]
            
            for step in range(horizon):
                if not active.any():
                    break
                
                # Forward pass for all active sequences
                logits = self.model({"input_ids": x, "padding_mask": pm})
                last_logits = logits[:, -1, :] / max(1e-8, self.config.temperature)
                
                # Top-k sampling
                if self.config.top_k > 0:
                    vals, idxs = torch.topk(last_logits, k=self.config.top_k, dim=-1)
                    probs = softmax(vals, dim=-1)
                    sampled_idx = torch.multinomial(probs, 1)
                    next_tokens = idxs.gather(-1, sampled_idx).squeeze(-1)
                else:
                    next_tokens = torch.argmax(last_logits, dim=-1)
                
                # Store tokens for active sequences
                for i in range(B):
                    if active[i]:
                        tid = int(next_tokens[i].item())
                        out_tokens[i].append(tid)
                        
                        # Check for DEATH token
                        if self.death_id is not None and tid == self.death_id:
                            active[i] = False
                
                # Extend sequences for next step (only if we have more steps)
                if step < horizon - 1 and active.any():
                    # Get last age and day from current sequences
                    last_ages = x[:, 1, -1]
                    last_days = x[:, 2, -1]
                    
                    # Build new step: (B, 4, 1)
                    new_step = torch.stack([
                        next_tokens,
                        last_ages,
                        last_days,
                        torch.ones(B, dtype=torch.long, device=self.device)
                    ], dim=1).unsqueeze(2)
                    
                    x = torch.cat([x, new_step], dim=2)
                    pm = torch.cat([pm, torch.ones(B, 1, dtype=pm.dtype, device=self.device)], dim=1)
            
            all_results.extend(out_tokens)
        
        return all_results
    
    @torch.no_grad()
    def _generate_tokens(
        self,
        prefix_4stream: torch.Tensor,
        pad_mask: torch.Tensor,
        horizon: int
    ) -> List[int]:
        """Generate tokens autoregressively (single sequence, for compatibility)."""
        result = self._generate_tokens_batch([prefix_4stream], [pad_mask], horizon, batch_size=1)
        return result[0]
    
    def _save_original_sequences(
        self,
        people_data: List[Dict],
        buddy_data: List[Dict],
        selected_indices: np.ndarray,
        buddy_indices: np.ndarray,
    ) -> str:
        """
        Save original sequences to Parquet file.
        
        Creates original_sequences.parquet with columns:
        - local_idx: 0 to n-1 for persons, n to 2n-1 for buddies
        - h5_idx: Original index in h5 file
        - rinpersoon_id: sequence_id from h5 file
        - original_sequence: Full token sequence from input_ids[idx, 0, :]
        """
        records = []
        
        # Add persons (indices 0 to n-1)
        for local_idx, person in enumerate(people_data):
            records.append({
                'local_idx': local_idx,
                'h5_idx': person['idx'],
                'rinpersoon_id': person['rinpersoon_id'],
                'original_sequence': ','.join(map(str, person['full_sequence'])),
                'real_length': person['L_real'],
                'is_buddy': False,
            })
        
        # Add buddies (indices n to 2n-1)
        n = len(people_data)
        for local_idx, buddy in enumerate(buddy_data):
            records.append({
                'local_idx': n + local_idx,
                'h5_idx': buddy['idx'],
                'rinpersoon_id': buddy['rinpersoon_id'],
                'original_sequence': ','.join(map(str, buddy['full_sequence'])),
                'real_length': buddy['L_real'],
                'is_buddy': True,
            })
        
        # Save to Parquet
        df = pd.DataFrame(records)
        original_path = os.path.join(self.config.output_dir, 'original_sequences.parquet')
        table = pa.Table.from_pandas(df)
        pq.write_table(table, original_path)
        
        logger.info(f"Saved original sequences: {original_path}")
        logger.info(f"  Persons: {len(people_data)}, Buddies: {len(buddy_data)}")
        
        return original_path
    
    def _save_ages(
        self,
        people_data: List[Dict],
        buddy_data: List[Dict],
        selected_indices: np.ndarray,
        buddy_indices: np.ndarray,
    ) -> str:
        """
        Save full age streams to Parquet file.
        
        Creates ages.parquet with columns:
        - local_idx: 0 to n-1 for persons, n to 2n-1 for buddies
        - h5_idx: Original index in h5 file
        - rinpersoon_id: sequence_id from h5 file
        - age_stream: Comma-separated ages for each position in the sequence
        - real_length: Real length of the sequence (before padding)
        - is_buddy: Whether this is a buddy sequence
        
        The age at any prefix_len p can be looked up as age_stream[p-1].
        This allows computing decade buckets that vary by prefix position.
        """
        records = []
        
        # Add persons (indices 0 to n-1)
        for local_idx, person in enumerate(people_data):
            records.append({
                'local_idx': local_idx,
                'h5_idx': person['idx'],
                'rinpersoon_id': person['rinpersoon_id'],
                'age_stream': ','.join(map(str, person['age_stream'])),
                'real_length': person['L_real'],
                'is_buddy': False,
            })
        
        # Add buddies (indices n to 2n-1)
        n = len(people_data)
        for local_idx, buddy in enumerate(buddy_data):
            records.append({
                'local_idx': n + local_idx,
                'h5_idx': buddy['idx'],
                'rinpersoon_id': buddy['rinpersoon_id'],
                'age_stream': ','.join(map(str, buddy['age_stream'])),
                'real_length': buddy['L_real'],
                'is_buddy': True,
            })
        
        # Save to Parquet
        df = pd.DataFrame(records)
        ages_path = os.path.join(self.config.output_dir, 'ages.parquet')
        table = pa.Table.from_pandas(df)
        pq.write_table(table, ages_path)
        
        # Log some statistics about ages for persons only
        persons_df = df[~df['is_buddy']]
        logger.info(f"Saved ages: {ages_path}")
        logger.info(f"  Total: {len(records)} (Persons: {len(people_data)}, Buddies: {len(buddy_data)})")
        logger.info(f"  Age streams stored for position-dependent decade lookup")
        
        return ages_path
    
    def generate(self):
        """Run sequence generation and save to Parquet files."""
        logger.info("="*60)
        logger.info(f"Starting Generation: {self.config.model_name}")
        logger.info(f"  exclude_padding: {self.config.exclude_padding}")
        logger.info("="*60)
        
        start_time = time.time()
        n = self.config.num_people
        c = self.config.num_generations
        h = self.config.horizon
        
        # Batch size for GPU generation
        # Adjust based on GPU memory - H100 can handle large batches
        generation_batch_size = self.config.generation_batch_size
        logger.info(f"Generation batch size: {generation_batch_size}")
        
        # Select people
        selected_indices, buddy_indices = self._select_people()
        logger.info(f"Selected {len(selected_indices)} people with buddies")
        
        # Load all person data
        logger.info("Loading person data...")
        people_data = [self._load_person_data(idx) for idx in tqdm(selected_indices, desc="Loading people")]
        buddy_data = [self._load_person_data(idx) for idx in tqdm(buddy_indices, desc="Loading buddies")]
        
        # Save original sequences first
        logger.info("Saving original sequences...")
        original_path = self._save_original_sequences(
            people_data, buddy_data, selected_indices, buddy_indices
        )
        
        # Save ages to separate file
        logger.info("Saving ages...")
        ages_path = self._save_ages(
            people_data, buddy_data, selected_indices, buddy_indices
        )
        
        # Storage for generation results
        records = []
        
        # Process each prefix length
        for prefix_len in tqdm(self.config.prefix_lengths, desc="Prefix lengths"):
            logger.info(f"Processing prefix_len={prefix_len}")
            
            # Collect all generation tasks for this prefix length
            # Each task = (person_idx, prefix_4stream, prefix_mask, buddy)
            generation_tasks = []
            task_metadata = []  # Store metadata for each task
            
            for person_idx, person in enumerate(people_data):
                # Skip if sequence too short
                if prefix_len + h > person['L_real']:
                    continue
                
                x4 = person['x4']
                pm = person['pm']
                buddy = buddy_data[person_idx]
                
                # Get prefix
                prefix_4stream = x4[:, :prefix_len]
                prefix_mask = pm[:prefix_len]
                
                # Add c copies of this prefix for c generations
                for gen_idx in range(c):
                    generation_tasks.append((prefix_4stream.clone(), prefix_mask.clone()))
                    task_metadata.append({
                        'person_idx': person_idx,
                        'rinpersoon_id': person['rinpersoon_id'],
                        'buddy_idx': person_idx,
                        'buddy_rinpersoon_id': buddy['rinpersoon_id'],
                        'prefix_len': prefix_len,
                        'generation_idx': gen_idx,
                    })
            
            if not generation_tasks:
                continue
            
            logger.info(f"  Generating {len(generation_tasks)} sequences in batches of {generation_batch_size}")
            
            # Batch generate all sequences
            prefixes = [t[0] for t in generation_tasks]
            masks = [t[1] for t in generation_tasks]
            
            generated_sequences = self._generate_tokens_batch(
                prefixes, masks, h, batch_size=generation_batch_size
            )
            
            # Store results - pad all sequences to exactly horizon length
            for i, (gen_tokens, meta) in enumerate(zip(generated_sequences, task_metadata)):
                # Pad to horizon length if shorter (e.g., if DEATH token was generated)
                if len(gen_tokens) < h:
                    pad_needed = h - len(gen_tokens)
                    gen_tokens = gen_tokens + [self.pad_id] * pad_needed
                
                records.append({
                    **meta,
                    'generated_tokens': ','.join(map(str, gen_tokens)),
                    'generated_len': len(gen_tokens),  # Always h after padding
                })
        
        logger.info(f"Generated {len(records)} records")
        
        # Save generated sequences to Parquet
        logger.info("Saving generated sequences...")
        df = pd.DataFrame(records)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self.config.sequences_path)
        
        file_size = os.path.getsize(self.config.sequences_path) / (1024 * 1024)
        elapsed = time.time() - start_time
        
        logger.info("="*60)
        logger.info(f"Generation Complete!")
        logger.info(f"  Original sequences: {original_path}")
        logger.info(f"  Generated sequences: {self.config.sequences_path}")
        logger.info(f"  Size: {file_size:.1f} MB")
        logger.info(f"  Records: {len(records)}")
        logger.info(f"  Time: {elapsed/60:.1f} minutes")
        logger.info("="*60)
        
        # Save metadata
        metadata = {
            'model_name': self.config.model_name,
            'num_people': n,
            'num_generations': c,
            'horizon': h,
            'prefix_lengths': self.config.prefix_lengths,
            'exclude_padding': self.config.exclude_padding,
            'vocab_size': self.vocab_size,
            'pad_id': self.pad_id,
            'max_seq_len': self.max_seq_len,
            'total_records': len(records),
            'selected_indices': selected_indices.tolist(),
            'buddy_indices': buddy_indices.tolist(),
            'elapsed_seconds': elapsed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        metadata_path = os.path.join(self.config.output_dir, 'generation_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return self.config.sequences_path


def load_config(config_path: str) -> GenerationConfig:
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    return GenerationConfig(
        model_name=cfg['model_name'],
        checkpoint_path=cfg['checkpoint_path'],
        data_path=cfg['data_path'],
        vocab_path=cfg['vocab_path'],
        output_dir=cfg['output_dir'],
        sequences_path=cfg['sequences_path'],
        num_people=cfg.get('num_people', 10),
        num_generations=cfg.get('num_generations', 100),
        horizon=cfg.get('horizon', 20),
        prefix_lengths=cfg.get('prefix_lengths'),
        exclude_padding=cfg.get('exclude_padding', True),
        generation_batch_size=cfg.get('generation_batch_size', 64),
        top_k=cfg.get('top_k', 20),
        temperature=cfg.get('temperature', 1.0),
        pad_token=cfg.get('pad_token', '[PAD]'),
        cls_token=cfg.get('cls_token', '[CLS]'),
        death_token=cfg.get('death_token', '[death]_[death]'),
        seed=cfg.get('seed', 42),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate sequences (GPU phase)")
    parser.add_argument("--config", required=True, help="Path to run config YAML")
    args = parser.parse_args()
    
    config = load_config(args.config)
    generator = SequenceGenerator(config)
    generator.generate()


if __name__ == "__main__":
    main()
