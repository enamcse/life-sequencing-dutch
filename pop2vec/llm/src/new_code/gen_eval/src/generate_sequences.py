#!/usr/bin/env python3
"""
Generate Sequences (GPU Phase)

Generates token sequences from prefixes and saves them to a single Parquet file.
This script is GPU-intensive and saves raw data for later statistical analysis.

Output: Single Parquet file with columns:
    - person_idx: Index of the person (0 to n-1)
    - prefix_len: Prefix length
    - generation_idx: Generation index (0 to c-1)
    - original_tokens: List of original continuation tokens
    - generated_tokens: List of generated tokens
    - buddy_tokens: Tokens from buddy (random pair)
    - next_tokens: Tokens from person (i+1) mod n

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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

from torch.nn.functional import softmax

# Project imports
from pop2vec.llm.src.new_code.utils import load_special_ids, load_vocab_df
from pop2vec.llm.src.new_code.load_data import CustomLazyHDF5Dataset
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
    
    # Sampling
    top_k: int = 20
    temperature: float = 1.0
    
    # Special tokens
    pad_token: str = "[PAD]"
    cls_token: str = "[CLS]"
    death_token: str = "[DEATH]"
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        if self.prefix_lengths is None:
            self.prefix_lengths = [1, 101, 201, 301, 401, 501, 601, 701, 801, 901, 1001]


class SequenceGenerator:
    """GPU-based sequence generator."""
    
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
        
        # Load model
        logger.info(f"Loading model: {config.checkpoint_path}")
        self.model = TransformerEncoder.load_from_checkpoint(
            config.checkpoint_path,
            strict=False
        )
        self.model.eval().to(self.device)
        
        # Load dataset
        logger.info(f"Loading dataset: {config.data_path}")
        self.dataset = CustomLazyHDF5Dataset(
            config.data_path,
            validation=False,
            num_val_items=100000,
            mlm_encoded=False,
            inference=True,
        )
        logger.info(f"Dataset size: {len(self.dataset)}")
    
    def _select_people(self) -> Tuple[np.ndarray, np.ndarray]:
        """Select n people and their random buddies."""
        n = self.config.num_people
        total = len(self.dataset)
        
        # Select main people
        selected = np.random.choice(total, size=min(n, total), replace=False)
        
        # Select random buddies (different from selected)
        buddies = np.random.choice(total, size=min(n, total), replace=False)
        for i in range(len(buddies)):
            while buddies[i] == selected[i]:
                buddies[i] = np.random.randint(0, total)
        
        return selected, buddies
    
    def _load_person_data(self, idx: int) -> Dict:
        """Load and preprocess a person's sequence."""
        item = self.dataset[idx]
        x4 = item["input_ids"]
        pm = item["padding_mask"]
        L_real = int(pm.sum().item())
        
        return {
            'idx': int(idx),
            'x4': x4[:, :L_real],
            'pm': pm[:L_real],
            'L_real': L_real,
            'rinpersoon_id': item.get('rinpersoon_id', int(idx)),
        }
    
    @torch.no_grad()
    def _generate_tokens(
        self,
        prefix_4stream: torch.Tensor,
        pad_mask: torch.Tensor,
        horizon: int
    ) -> List[int]:
        """Generate tokens autoregressively."""
        x = prefix_4stream.unsqueeze(0).to(self.device)
        pm = pad_mask.unsqueeze(0).to(self.device)
        out_tokens = []
        
        for _ in range(horizon):
            logits = self.model({"input_ids": x, "padding_mask": pm})
            last_logits = logits[:, -1, :] / max(1e-8, self.config.temperature)
            
            if self.config.top_k > 0:
                vals, idxs = torch.topk(last_logits, k=self.config.top_k, dim=-1)
                probs = softmax(vals, dim=-1)
                next_token = idxs.gather(-1, torch.multinomial(probs, 1)).squeeze(-1)
            else:
                next_token = torch.argmax(last_logits, dim=-1)
            
            tid = int(next_token.item())
            out_tokens.append(tid)
            
            if self.death_id is not None and tid == self.death_id:
                break
            
            # Extend sequence
            last_age = x[0, 1, -1].item()
            last_day = x[0, 2, -1].item()
            new_step = torch.tensor([[tid], [last_age], [last_day], [1]],
                                   dtype=torch.long, device=self.device)
            x = torch.cat([x, new_step.unsqueeze(0)], dim=2)
            pm = torch.cat([pm, torch.ones(1, 1, dtype=pm.dtype, device=self.device)], dim=1)
        
        return out_tokens
    
    def generate(self):
        """Run sequence generation and save to Parquet."""
        logger.info("="*60)
        logger.info(f"Starting Generation: {self.config.model_name}")
        logger.info("="*60)
        
        start_time = time.time()
        n = self.config.num_people
        c = self.config.num_generations
        h = self.config.horizon
        
        # Select people
        selected_indices, buddy_indices = self._select_people()
        logger.info(f"Selected {len(selected_indices)} people with buddies")
        
        # Load all person data
        logger.info("Loading person data...")
        people_data = [self._load_person_data(idx) for idx in tqdm(selected_indices, desc="Loading people")]
        buddy_data = [self._load_person_data(idx) for idx in tqdm(buddy_indices, desc="Loading buddies")]
        
        # Storage for results
        records = []
        
        # Process each prefix length
        for prefix_len in tqdm(self.config.prefix_lengths, desc="Prefix lengths"):
            logger.info(f"Processing prefix_len={prefix_len}")
            
            for person_idx, person in enumerate(tqdm(people_data, desc=f"Prefix {prefix_len}", leave=False)):
                # Skip if sequence too short
                if prefix_len + h > person['L_real']:
                    continue
                
                x4 = person['x4']
                pm = person['pm']
                
                # Get prefix
                prefix_4stream = x4[:, :prefix_len]
                prefix_mask = pm[:prefix_len]
                
                # Get original continuation
                original_tokens = x4[0, prefix_len:prefix_len+h].tolist()
                
                # Get buddy's continuation
                buddy = buddy_data[person_idx]
                if prefix_len + h <= buddy['L_real']:
                    buddy_tokens = buddy['x4'][0, prefix_len:prefix_len+h].tolist()
                else:
                    buddy_tokens = buddy['x4'][0, prefix_len:buddy['L_real']].tolist()
                
                # Get next person's continuation (circular)
                next_person = people_data[(person_idx + 1) % n]
                if prefix_len + h <= next_person['L_real']:
                    next_tokens = next_person['x4'][0, prefix_len:prefix_len+h].tolist()
                else:
                    next_tokens = next_person['x4'][0, prefix_len:next_person['L_real']].tolist()
                
                # Generate c times
                for gen_idx in range(c):
                    generated_tokens = self._generate_tokens(prefix_4stream, prefix_mask, h)
                    
                    records.append({
                        'person_idx': person_idx,
                        'rinpersoon_id': person['rinpersoon_id'],
                        'prefix_len': prefix_len,
                        'generation_idx': gen_idx,
                        'original_tokens': original_tokens,
                        'generated_tokens': generated_tokens,
                        'buddy_tokens': buddy_tokens,
                        'next_tokens': next_tokens,
                        'original_len': len(original_tokens),
                        'generated_len': len(generated_tokens),
                        'buddy_len': len(buddy_tokens),
                        'next_len': len(next_tokens),
                    })
        
        logger.info(f"Generated {len(records)} records")
        
        # Convert to Parquet
        # Note: Lists are stored as strings for Parquet compatibility
        logger.info("Converting to Parquet...")
        
        parquet_records = []
        for r in records:
            parquet_records.append({
                'person_idx': r['person_idx'],
                'rinpersoon_id': r['rinpersoon_id'],
                'prefix_len': r['prefix_len'],
                'generation_idx': r['generation_idx'],
                'original_tokens': ','.join(map(str, r['original_tokens'])),
                'generated_tokens': ','.join(map(str, r['generated_tokens'])),
                'buddy_tokens': ','.join(map(str, r['buddy_tokens'])),
                'next_tokens': ','.join(map(str, r['next_tokens'])),
                'original_len': r['original_len'],
                'generated_len': r['generated_len'],
                'buddy_len': r['buddy_len'],
                'next_len': r['next_len'],
            })
        
        # Create PyArrow table
        import pandas as pd
        df = pd.DataFrame(parquet_records)
        table = pa.Table.from_pandas(df)
        
        # Write Parquet
        pq.write_table(table, self.config.sequences_path)
        
        file_size = os.path.getsize(self.config.sequences_path) / (1024 * 1024)
        elapsed = time.time() - start_time
        
        logger.info("="*60)
        logger.info(f"Generation Complete!")
        logger.info(f"  Output: {self.config.sequences_path}")
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
            'vocab_size': self.vocab_size,
            'pad_id': self.pad_id,
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
        top_k=cfg.get('top_k', 20),
        temperature=cfg.get('temperature', 1.0),
        pad_token=cfg.get('pad_token', '[PAD]'),
        cls_token=cfg.get('cls_token', '[CLS]'),
        death_token=cfg.get('death_token', '[DEATH]'),
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
