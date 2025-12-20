#!/usr/bin/env python3
"""
Script to add birthday tokens to preprocessed life sequence data.
Reads from existing HDF5 file, inserts BIRTHDAY_YEAR_X tokens, and saves to new file.
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Import optimized dataset class for preprocessing
from pop2vec.llm.src.new_code.load_data import PreprocessingLazyHDF5Dataset

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class BirthdayTokenInserter:
    """Handles birthday token insertion logic"""
    
    def __init__(self, vocab_path: str, max_seq_len: int = 512):
        """
        Args:
            vocab_path: Path to vocabulary CSV file
            max_seq_len: Maximum sequence length after insertion
        """
        self.max_seq_len = max_seq_len
        self.vocab_df = pd.read_csv(vocab_path)
        
        # Create mapping for quick lookups
        if 'token' in self.vocab_df.columns:
            # lowercase column names
            self.token_to_id = dict(zip(self.vocab_df['token'], self.vocab_df['id']))
            self.id_to_token = dict(zip(self.vocab_df['id'], self.vocab_df['token']))
        else:
            # uppercase column names (most likely case)
            self.token_to_id = dict(zip(self.vocab_df['TOKEN'], self.vocab_df['ID']))
            self.id_to_token = dict(zip(self.vocab_df['ID'], self.vocab_df['TOKEN']))
        
        # Get birthday token IDs (will be added if missing)
        self.birthday_token_ids = self._ensure_birthday_tokens()
        
        # Important token IDs
        self.sep_id = self.token_to_id.get('[SEP]', 2)
        self.cls_id = self.token_to_id.get('[CLS]', 1)
        self.pad_id = self.token_to_id.get('[PAD]', 0)
        
        logger.info(f"Loaded vocabulary with {len(self.vocab_df)} tokens")
        logger.info(f"Birthday tokens: {len(self.birthday_token_ids)} (age 1-100)")
    
    def _ensure_birthday_tokens(self) -> Dict[int, int]:
        """Initialize birthday token tracking. Tokens will be added on-demand."""
        # Use appropriate column names
        self.token_col = 'TOKEN' if 'TOKEN' in self.vocab_df.columns else 'token'
        self.id_col = 'ID' if 'ID' in self.vocab_df.columns else 'id'
        self.category_col = 'CATEGORY' if 'CATEGORY' in self.vocab_df.columns else 'category'
        
        # Find existing birthday tokens
        birthday_tokens = {}
        for _, row in self.vocab_df.iterrows():
            token_name = row[self.token_col]
            if isinstance(token_name, str) and token_name.startswith("BIRTHDAY_YEAR_"):
                try:
                    age = int(token_name.split("_")[-1])
                    birthday_tokens[age] = int(row[self.id_col])
                except ValueError:
                    continue
        
        logger.info(f"Found {len(birthday_tokens)} existing birthday tokens in vocabulary")
        return birthday_tokens
    
    def _add_birthday_token(self, age: int) -> int:
        """Add a single birthday token to vocabulary on-demand. Returns token ID."""
        token_name = f"BIRTHDAY_YEAR_{age}"
        
        if token_name in self.token_to_id:
            return self.token_to_id[token_name]
        
        # Add new token
        new_id = len(self.vocab_df)
        new_row = {
            self.token_col: token_name,
            self.id_col: new_id,
            self.category_col: 'TEMPORAL'
        }
        
        # Add to dataframe
        new_df = pd.DataFrame([new_row])
        self.vocab_df = pd.concat([self.vocab_df, new_df], ignore_index=True)
        
        # Update mappings
        self.token_to_id[token_name] = new_id
        self.id_to_token[new_id] = token_name
        self.birthday_token_ids[age] = new_id
        
        logger.info(f"Added birthday token for age {age}: {token_name} -> ID {new_id}")
        return new_id
    
    def save_updated_vocabulary(self, output_path: Optional[str] = None):
        """Save the updated vocabulary with any new birthday tokens."""
        if output_path is None:
            output_path = "vocab_with_birthdays.csv"
        
        self.vocab_df.to_csv(output_path, index=False)
        logger.info(f"Saved updated vocabulary with {len(self.vocab_df)} tokens to {output_path}")
    
    def insert_birthdays(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Insert birthday tokens based on age gaps in the sequence.
        
        Args:
            sample: Dictionary containing 'input_ids' and 'padding_mask'
            
        Returns:
            Modified sample with birthday tokens inserted
        """
        input_ids = sample["input_ids"].clone()  # (4, L)
        padding_mask = sample["padding_mask"].clone()  # (L,)
        
        # Get real sequence length
        real_len = int(padding_mask.sum().item())
        if real_len < 2:
            return sample  # Skip very short sequences
            
        # Extract real sequence (no padding)
        real_input_ids = input_ids[:, :real_len]  # (4, L_real)
        
        # Insert birthday tokens based on age gaps
        new_input_ids = self._insert_birthday_tokens_by_age_gaps(real_input_ids)
        
        # Apply truncation (preserve first 6 demographic tokens, truncate from the middle)
        if new_input_ids.size(1) > self.max_seq_len:
            # Keep first 6 tokens (demographic header: CLS, municipality, gender, month, year, SEP)
            demographic_header = new_input_ids[:, :6]
            # Keep most recent tokens from the rest
            remaining_space = self.max_seq_len - 6
            recent_tokens = new_input_ids[:, -remaining_space:]
            new_input_ids = torch.cat([demographic_header, recent_tokens], dim=1)
        
        # Create new padding mask
        new_real_len = new_input_ids.size(1)
        new_padding_mask = torch.zeros(self.max_seq_len, dtype=padding_mask.dtype)
        new_padding_mask[:new_real_len] = 1
        
        # Pad input_ids to max_seq_len
        if new_input_ids.size(1) < self.max_seq_len:
            pad_size = self.max_seq_len - new_input_ids.size(1)
            padding = torch.zeros(4, pad_size, dtype=input_ids.dtype)
            new_input_ids = torch.cat([new_input_ids, padding], dim=1)
        
        # Update sample
        sample["input_ids"] = new_input_ids
        sample["padding_mask"] = new_padding_mask
        
        return sample
    
    def _extract_birth_info(self, background_tokens: torch.Tensor) -> Tuple[int, int]:
        """
        Extract birth year and month from background tokens.
        Expected format: [CLS], municipality_X, gender_X, month_X, year_YYYY, [SEP]
        
        Returns:
            (birth_year, birth_month) tuple
        """
        birth_year = None
        birth_month = None
        
        for token_id in background_tokens:
            token_id = int(token_id)
            if token_id in self.id_to_token:
                token_name = self.id_to_token[token_id]
                if isinstance(token_name, str):
                    if token_name.startswith("month_"):
                        try:
                            birth_month = int(token_name.split("_")[1])
                        except (ValueError, IndexError):
                            pass
                    elif token_name.startswith("year_"):
                        try:
                            birth_year = int(token_name.split("_")[1])
                        except (ValueError, IndexError):
                            pass
        
        # Use defaults if not found
        if birth_year is None:
            logger.warning("Birth year not found in background, using 1970")
            birth_year = 1970
        if birth_month is None:
            logger.warning("Birth month not found in background, using 1")
            birth_month = 1
            
        return birth_year, birth_month
    
    def _calculate_genesis_days(self, birth_year: int, birth_month: int, genesis_year: int = 1970) -> int:
        """
        Calculate days from genesis date to birth date.
        Genesis date is typically 1970-01-01 (Unix epoch).
        
        Args:
            birth_year: Person's birth year
            birth_month: Person's birth month (1-12)
            genesis_year: Reference year (default 1970)
            
        Returns:
            Number of days from genesis_year-01-01 to birth date
        """
        from datetime import datetime
        
        genesis_date = datetime(genesis_year, 1, 1)
        birth_date = datetime(birth_year, birth_month, 1)  # Use 1st day of birth month
        
        days_diff = (birth_date - genesis_date).days
        return days_diff
    
    def _calculate_birthday_date(self, genesis_days: int, age: int) -> int:
        """
        Calculate the absolute date (in days) for a birthday at given age.
        
        Args:
            genesis_days: Days from genesis to birth
            age: Age at birthday
            
        Returns:
            Absolute date in days from genesis
        """
        # Approximate: each year = 365.25 days (accounts for leap years)
        days_since_birth = int(age * 365.25)
        return genesis_days + days_since_birth
    
    def _insert_birthday_tokens_by_age_gaps(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Insert birthday tokens based on age gaps in the sequence"""
        tokens = input_ids[0]  # (L,)
        abspos = input_ids[1]  # (L,)
        ages = input_ids[2]    # (L,)
        segments = input_ids[3]  # (L,)
        
        # Find where background ends (first [SEP])
        sep_positions = (tokens == self.sep_id).nonzero(as_tuple=False)
        if len(sep_positions) == 0:
            return input_ids  # No [SEP] found
            
        bg_end = int(sep_positions[0].item())
        
        # Extract birth year and month from background tokens for date calculation
        birth_year, birth_month = self._extract_birth_info(tokens[:bg_end+1])
        genesis_date_days = self._calculate_genesis_days(birth_year, birth_month)
        
        # Check for death tokens - stop processing if found
        death_token_id = self.token_to_id.get('DEATH', None)
        death_positions = []
        if death_token_id is not None:
            death_positions = (tokens == death_token_id).nonzero(as_tuple=False)
        
        # Process sequence after background
        new_events = []
        
        # Add background (unchanged)
        for i in range(bg_end + 1):
            new_events.append({
                'token': int(tokens[i]),
                'abspos': int(abspos[i]),
                'age': int(ages[i]),
                'segment': int(segments[i])
            })
        
        # Track last age seen (skip age 0)
        last_age = 0
        
        # Process tokens after background
        for i in range(bg_end + 1, len(tokens)):
            if tokens[i] == self.pad_id:
                break  # Stop at padding
                
            # Stop if we hit a death token
            if death_token_id is not None and tokens[i] == death_token_id:
                # Add the death token and stop
                new_events.append({
                    'token': int(tokens[i]),
                    'abspos': int(abspos[i]),
                    'age': int(ages[i]),
                    'segment': int(segments[i])
                })
                break
                
            current_age = int(ages[i])
            
            # Skip age 0 tokens
            if current_age == 0:
                new_events.append({
                    'token': int(tokens[i]),
                    'abspos': int(abspos[i]),
                    'age': int(ages[i]),
                    'segment': int(segments[i])
                })
                continue
            
            # If we have an age gap, insert birthday tokens
            if current_age > last_age + 1:
                # Insert birthday tokens for missing ages
                for missing_age in range(last_age + 1, current_age):
                    # Create birthday token if it doesn't exist
                    if missing_age not in self.birthday_token_ids:
                        self._add_birthday_token(missing_age)
                    
                    # Calculate correct birthday date
                    birthday_date = self._calculate_birthday_date(genesis_date_days, missing_age)
                    
                    # Add birthday token
                    new_events.append({
                        'token': self.birthday_token_ids[missing_age],
                        'abspos': birthday_date,  # Use calculated date
                        'age': missing_age,
                        'segment': 1  # Temporal segment
                    })
                    
                    # Add [SEP] after birthday
                    new_events.append({
                        'token': self.sep_id,
                        'abspos': birthday_date,  # Same date as birthday
                        'age': missing_age,
                        'segment': 1
                    })
            
            # Add the current event
            new_events.append({
                'token': int(tokens[i]),
                'abspos': int(abspos[i]),
                'age': int(ages[i]),
                'segment': int(segments[i])
            })
            
            # Update last age seen
            if current_age > 0:
                last_age = current_age
        
        # Convert back to tensor format
        new_len = len(new_events)
        new_input_ids = torch.zeros(4, new_len, dtype=input_ids.dtype)
        
        for i, event in enumerate(new_events):
            new_input_ids[0, i] = event['token']
            new_input_ids[1, i] = event['abspos']
            new_input_ids[2, i] = event['age']
            new_input_ids[3, i] = event['segment']
        
        return new_input_ids


def process_batch_parallel(batch_indices, input_path, inserter_config, max_seq_len, mlm_encoded=False):
    """
    Process a batch of samples in parallel worker.
    This function will be called by each worker process.
    """
    # Create inserter in each worker (can't pickle the full object)
    vocab_path, birthday_token_ids, sep_id, cls_id, pad_id = inserter_config
    
    # Load vocabulary in worker
    vocab_df = pd.read_csv(vocab_path)
    
    # Handle both uppercase and lowercase column names
    if 'TOKEN' in vocab_df.columns:
        token_to_id = dict(zip(vocab_df['TOKEN'], vocab_df['ID']))
        id_to_token = dict(zip(vocab_df['ID'], vocab_df['TOKEN']))
    else:
        token_to_id = dict(zip(vocab_df['token'], vocab_df['id']))
        id_to_token = dict(zip(vocab_df['id'], vocab_df['token']))
    
    # Create mini-inserter for this worker
    class WorkerInserter:
        def __init__(self):
            self.birthday_token_ids = birthday_token_ids
            self.sep_id = sep_id
            self.cls_id = cls_id 
            self.pad_id = pad_id
            self.token_to_id = token_to_id
            self.id_to_token = id_to_token
            self.max_seq_len = max_seq_len
        
        def _extract_birth_info(self, background_tokens):
            """Extract birth year and month from background tokens"""
            birth_year = None
            birth_month = None
            
            for token_id in background_tokens:
                token_id = int(token_id)
                if token_id in self.id_to_token:
                    token_name = self.id_to_token[token_id]
                    if isinstance(token_name, str):
                        if token_name.startswith("month_"):
                            try:
                                birth_month = int(token_name.split("_")[1])
                            except (ValueError, IndexError):
                                pass
                        elif token_name.startswith("year_"):
                            try:
                                birth_year = int(token_name.split("_")[1])
                            except (ValueError, IndexError):
                                pass
            
            return (birth_year or 1970, birth_month or 1)
        
        def _calculate_genesis_days(self, birth_year, birth_month, genesis_year=1970):
            """Calculate days from genesis date to birth date"""
            from datetime import datetime
            genesis_date = datetime(genesis_year, 1, 1)
            birth_date = datetime(birth_year, birth_month, 1)
            return (birth_date - genesis_date).days
        
        def _calculate_birthday_date(self, genesis_days, age):
            """Calculate the absolute date for a birthday at given age"""
            days_since_birth = int(age * 365.25)
            return genesis_days + days_since_birth
        
        def _insert_birthday_tokens_by_age_gaps(self, input_ids):
            tokens = input_ids[0]
            abspos = input_ids[1]
            ages = input_ids[2]
            segments = input_ids[3]
            
            sep_positions = (tokens == self.sep_id).nonzero(as_tuple=False)
            if len(sep_positions) == 0:
                return input_ids
                
            bg_end = int(sep_positions[0].item())
            
            # Extract birth info and calculate genesis days
            birth_year, birth_month = self._extract_birth_info(tokens[:bg_end+1])
            genesis_date_days = self._calculate_genesis_days(birth_year, birth_month)
            
            # Check for death tokens
            death_token_id = self.token_to_id.get('DEATH', None)
            
            new_events = []
            
            # Add background (unchanged)
            for i in range(bg_end + 1):
                new_events.append({
                    'token': int(tokens[i]),
                    'abspos': int(abspos[i]),
                    'age': int(ages[i]),
                    'segment': int(segments[i])
                })
            
            # Track last age seen (skip age 0)
            last_age = 0
            
            # Process tokens after background
            for i in range(bg_end + 1, len(tokens)):
                if tokens[i] == self.pad_id:
                    break  # Stop at padding
                    
                # Stop if we hit a death token
                if death_token_id is not None and tokens[i] == death_token_id:
                    new_events.append({
                        'token': int(tokens[i]),
                        'abspos': int(abspos[i]),
                        'age': int(ages[i]),
                        'segment': int(segments[i])
                    })
                    break
                    
                current_age = int(ages[i])
                
                # Skip age 0 tokens
                if current_age == 0:
                    new_events.append({
                        'token': int(tokens[i]),
                        'abspos': int(abspos[i]),
                        'age': int(ages[i]),
                        'segment': int(segments[i])
                    })
                    continue
                
                # If we have an age gap, insert birthday tokens
                if current_age > last_age + 1:
                    # Insert birthday tokens for missing ages
                    num_inserted = 0
                    for missing_age in range(last_age + 1, current_age):
                        # Only add if birthday token exists (pre-populated)
                        if missing_age in self.birthday_token_ids:
                            # Calculate correct birthday date
                            birthday_date = self._calculate_birthday_date(genesis_date_days, missing_age)
                            
                            new_events.append({
                                'token': self.birthday_token_ids[missing_age],
                                'abspos': birthday_date,
                                'age': missing_age,
                                'segment': 1
                            })
                            
                            new_events.append({
                                'token': self.sep_id,
                                'abspos': birthday_date,
                                'age': missing_age,
                                'segment': 1
                            })
                            num_inserted += 1
                
                # Add the current event
                new_events.append({
                    'token': int(tokens[i]),
                    'abspos': int(abspos[i]),
                    'age': int(ages[i]),
                    'segment': int(segments[i])
                })
                
                # Update last age seen
                if current_age > 0:
                    last_age = current_age
            
            # Convert back to tensor format
            new_len = len(new_events)
            new_input_ids = torch.zeros(4, new_len, dtype=input_ids.dtype)
            
            for i, event in enumerate(new_events):
                new_input_ids[0, i] = event['token']
                new_input_ids[1, i] = event['abspos']
                new_input_ids[2, i] = event['age']
                new_input_ids[3, i] = event['segment']
            
            return new_input_ids
        
        def insert_birthdays(self, sample):
            input_ids = sample["input_ids"].clone()
            padding_mask = sample["padding_mask"].clone()
            
            real_len = int(padding_mask.sum().item())
            if real_len < 2:
                return sample
                
            real_input_ids = input_ids[:, :real_len]
            
            new_input_ids = self._insert_birthday_tokens_by_age_gaps(real_input_ids)
            
            if new_input_ids.size(1) > self.max_seq_len:
                # Keep first 6 tokens (demographic header: CLS, municipality, gender, month, year, SEP)
                demographic_header = new_input_ids[:, :6]
                # Keep most recent tokens from the rest
                remaining_space = self.max_seq_len - 6
                recent_tokens = new_input_ids[:, -remaining_space:]
                new_input_ids = torch.cat([demographic_header, recent_tokens], dim=1)
            
            new_real_len = new_input_ids.size(1)
            new_padding_mask = torch.zeros(self.max_seq_len, dtype=padding_mask.dtype)
            new_padding_mask[:new_real_len] = 1
            
            if new_input_ids.size(1) < self.max_seq_len:
                pad_size = self.max_seq_len - new_input_ids.size(1)
                padding = torch.zeros(4, pad_size, dtype=input_ids.dtype)
                new_input_ids = torch.cat([new_input_ids, padding], dim=1)
            
            sample["input_ids"] = new_input_ids
            sample["padding_mask"] = new_padding_mask
            
            return sample
    
    # Create dataset for this worker (each worker gets its own HDF5 handle)
    dataset = PreprocessingLazyHDF5Dataset(
        input_path,
        inference=True,
        mlm_encoded=mlm_encoded,  # Use configurable value
        return_index=True
    )
    
    # Create worker inserter
    inserter = WorkerInserter()
    
    # Process batch
    processed_batch = []
    for idx in batch_indices:
        try:
            sample = dataset[idx]
            
            # CRITICAL: Convert tensors to numpy immediately to prevent shared memory issues
            sample_detached = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    # Force to CPU numpy to break shared memory connection
                    sample_detached[key] = value.detach().cpu().numpy()
                else:
                    sample_detached[key] = value
            
            # Convert back to new tensors (not shared)
            sample_new = {}
            for key, value in sample_detached.items():
                if isinstance(value, np.ndarray):
                    sample_new[key] = torch.from_numpy(value.copy())  # .copy() prevents sharing
                else:
                    sample_new[key] = value
            
            processed_sample = inserter.insert_birthdays(sample_new)
            
            # Convert result to numpy for safe return
            result_sample = {}
            for key, value in processed_sample.items():
                if isinstance(value, torch.Tensor):
                    result_sample[key] = value.detach().cpu().numpy()
                else:
                    result_sample[key] = value
            
            processed_batch.append((idx, result_sample))
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # Skip failed samples to prevent entire batch failure
            continue
    
    return processed_batch


def process_file(
    input_path: str,
    output_path: str,
    vocab_path: str,
    max_seq_len: int = 512,
    batch_size: int = 1000,
    num_workers: int = None,
    mlm_encoded: bool = False,
    device: torch.device = torch.device("cpu")
):
    """
    Process an HDF5 file to add birthday tokens using multiprocessing.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
        vocab_path: Path to vocabulary CSV file
        max_seq_len: Maximum sequence length
        batch_size: Number of samples per batch
        num_workers: Number of parallel workers (default: CPU count)
        mlm_encoded: Whether the HDF5 contains MLM data (default: False for generative models)
        device: PyTorch device for tensor operations (default: CPU)
    """
    if num_workers is None:
        if device.type == "cuda":
            # For GPU: Use fewer workers (one per GPU typically)
            num_workers = 4  # Suitable for 2-4 GPU nodes
        else:
            # For CPU: Use many workers for parallel processing
            num_workers = min(cpu_count(), 32)  # Cap at 32 to avoid too many processes
    else:
        # User specified num_workers, but adjust for device type
        if device.type == "cuda":
            # Limit GPU workers to reasonable number
            original_workers = num_workers
            num_workers = min(num_workers, 4)
            logger.info(f"GPU detected: Limiting workers to {num_workers} (user requested {original_workers})")
        # For CPU, use whatever user specified
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("STARTING BIRTHDAY TOKEN INSERTION")
    logger.info("=" * 80)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Vocab:  {vocab_path}")
    logger.info("")
    logger.info("CONFIGURATION:")
    logger.info(f"  max_seq_len:  {max_seq_len}")
    logger.info(f"  batch_size:   {batch_size}")
    logger.info(f"  num_workers:  {num_workers}")
    logger.info(f"  mlm_encoded:  {mlm_encoded}")
    logger.info(f"  device:       {device}")
    logger.info("=" * 80)
    
    # Step 1: Scan all data to find unique ages (pre-processing step)
    logger.info("Step 1: Scanning data to find all unique ages...")
    dataset = PreprocessingLazyHDF5Dataset(
        input_path,
        inference=True,
        mlm_encoded=mlm_encoded,
        return_index=True
    )
    
    # Sample a subset to find age ranges (for efficiency)
    sample_size = min(1000, len(dataset))
    unique_ages = set()
    
    for i in range(0, len(dataset), len(dataset) // sample_size + 1):
        sample = dataset[i]
        input_ids = sample["input_ids"]
        padding_mask = sample["padding_mask"]
        real_len = int(padding_mask.sum().item())
        
        if real_len > 1:
            ages = input_ids[2, :real_len]  # Age stream
            unique_ages.update(ages[ages > 0].cpu().numpy().tolist())
    
    max_age = max(unique_ages) if unique_ages else 100
    logger.info(f"Found age range in data: {min(unique_ages) if unique_ages else 0} to {max_age}")
    
    # Step 2: Pre-populate vocabulary with ALL birthday tokens from age 1 to max_age
    # We need tokens for ALL ages, not just those present in the data,
    # because we'll insert birthdays for missing age gaps
    inserter = BirthdayTokenInserter(vocab_path, max_seq_len)
    for age in range(1, min(max_age + 1, 101)):  # Ages 1-100 (or up to max_age)
        inserter._add_birthday_token(age)
    
    logger.info(f"Pre-populated {min(max_age, 100)} birthday tokens in vocabulary (age 1-{min(max_age, 100)})")
    
    # Step 3: Process data with pre-populated vocabulary
    # Create config for workers (can't pickle the full inserter)
    inserter_config = (
        vocab_path,
        inserter.birthday_token_ids,
        inserter.sep_id,
        inserter.cls_id,
        inserter.pad_id
    )
    
    n_samples = len(dataset)
    logger.info(f"Dataset contains {n_samples:,} samples")
    
    # Determine what fields to copy
    sample = dataset[0]
    fields = list(sample.keys())
    
    # Create output file
    os.makedirs(Path(output_path).parent, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f_out:
        # Pre-allocate datasets
        f_out.create_dataset('input_ids', (n_samples, 4, max_seq_len), dtype=np.int64)
        f_out.create_dataset('padding_mask', (n_samples, max_seq_len), dtype=np.int64)
        
        # Copy other fields as-is
        if 'sequence_id' in fields:
            f_out.create_dataset('sequence_id', (n_samples,), dtype=np.int64)
        if 'original_sequence' in fields:
            f_out.create_dataset('original_sequence', (n_samples, max_seq_len), dtype=np.int64)
        if 'target_tokens' in fields and mlm_encoded:
            # These need special handling for variable length (only for MLM data)
            with h5py.File(input_path, 'r') as f_in:
                f_out.create_dataset('target_tokens', data=f_in['target_tokens'][:])
                f_out.create_dataset('target_pos', data=f_in['target_pos'][:])
                f_out.create_dataset('target_cls', data=f_in['target_cls'][:])
        
        # Create batches of indices for parallel processing
        batch_indices = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices.append(list(range(start_idx, end_idx)))
        
        logger.info(f"Processing {len(batch_indices)} batches with {num_workers} workers...")
        
        # Process batches in parallel
        process_func = partial(
            process_batch_parallel,
            input_path=input_path,
            inserter_config=inserter_config,
            max_seq_len=max_seq_len,
            mlm_encoded=mlm_encoded
        )
        
        with Pool(processes=num_workers) as pool:
            # Process all batches in parallel with progress bar
            results = []
            for batch_result in tqdm(
                pool.imap(process_func, batch_indices),
                total=len(batch_indices),
                desc="Processing batches"
            ):
                results.extend(batch_result)
        
        # Sort results by index to maintain order
        results.sort(key=lambda x: x[0])
        
        # Write results to output file
        logger.info("Writing results to output file...")
        
        # Track statistics
        total_birthday_tokens_inserted = 0
        total_sequence_length_before = 0
        total_sequence_length_after = 0
        num_sequences_modified = 0
        
        for idx, (original_idx, sample) in enumerate(tqdm(results, desc="Writing samples")):
            # Write modified fields - handle both tensor and numpy array inputs
            input_ids = sample['input_ids']
            padding_mask = sample['padding_mask']
            
            # Convert to numpy if it's a tensor, otherwise use as-is
            if hasattr(input_ids, 'numpy'):
                input_ids = input_ids.numpy()
            if hasattr(padding_mask, 'numpy'):
                padding_mask = padding_mask.numpy()
            
            # Count birthday tokens in this sequence
            tokens = input_ids[0]
            birthday_count = sum(1 for token_id in tokens if inserter.id_to_token.get(int(token_id), '').startswith('BIRTHDAY_YEAR_'))
            total_birthday_tokens_inserted += birthday_count
            if birthday_count > 0:
                num_sequences_modified += 1
            
            # Track sequence length changes
            seq_len = int(padding_mask.sum())
            total_sequence_length_after += seq_len
                
            f_out['input_ids'][original_idx] = input_ids
            f_out['padding_mask'][original_idx] = padding_mask
            
            # Write unchanged fields
            if 'sequence_id' in sample:
                seq_id = sample['sequence_id']
                if hasattr(seq_id, 'numpy'):
                    seq_id = seq_id.numpy()
                f_out['sequence_id'][original_idx] = seq_id
            if 'original_sequence' in sample:
                # Update original_sequence to match new input_ids[0]
                orig_seq = sample['input_ids'][0]
                if hasattr(orig_seq, 'numpy'):
                    orig_seq = orig_seq.numpy()
                f_out['original_sequence'][original_idx] = orig_seq
    
    # Print comprehensive summary
    logger.info("=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Input file:  {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info("")
    logger.info("BIRTHDAY TOKEN INSERTION STATISTICS:")
    logger.info(f"  Total samples processed:         {n_samples:,}")
    logger.info(f"  Sequences modified:              {num_sequences_modified:,} ({100*num_sequences_modified/n_samples:.1f}%)")
    logger.info(f"  Sequences unchanged:             {n_samples - num_sequences_modified:,} ({100*(n_samples - num_sequences_modified)/n_samples:.1f}%)")
    logger.info("")
    logger.info(f"  Total birthday tokens inserted:  {total_birthday_tokens_inserted:,}")
    logger.info(f"  Average per sequence (all):      {total_birthday_tokens_inserted/n_samples:.2f}")
    if num_sequences_modified > 0:
        logger.info(f"  Average per modified sequence:   {total_birthday_tokens_inserted/num_sequences_modified:.2f}")
    logger.info("")
    logger.info(f"  Average sequence length (after): {total_sequence_length_after/n_samples:.1f} tokens")
    logger.info(f"  Maximum sequence length:         {max_seq_len}")
    logger.info("")
    logger.info("VOCABULARY:")
    logger.info(f"  Birthday tokens in vocabulary:   {len(inserter.birthday_token_ids)} (age 1-{max(inserter.birthday_token_ids.keys())})")
    logger.info(f"  Total vocabulary size:           {len(inserter.vocab_df):,} tokens")
    logger.info("=" * 80)
    
    # Save updated vocabulary with any new birthday tokens
    vocab_output_path = Path(vocab_path).parent / "vocab_with_birthdays.csv"
    inserter.save_updated_vocabulary(str(vocab_output_path))


def main():
    parser = argparse.ArgumentParser(description="Add birthday tokens to life sequence data")
    parser.add_argument("config", help="JSON config file path")
    
    args = parser.parse_args()
    
    # Print to stdout (goes to .out file) - keep minimal
    print(f"Starting birthday token insertion job...")
    print(f"Config: {args.config}")
    
    # Load configuration
    with open(args.config) as f:
        config = json.load(f)
    
    # Required config fields
    input_path = config["input_file"]      # Path to input HDF5
    output_path = config["output_file"]    # Path to output HDF5
    vocab_path = config["vocab_file"]      # Path to vocabulary CSV
    
    # Optional config fields
    max_seq_len = config.get("max_seq_len", 512)
    batch_size = config.get("batch_size", 1000)
    num_workers = config.get("num_workers", None)
    mlm_encoded = config.get("mlm_encoded", False)  # Default False for generative models
    
    # GPU configuration
    use_gpu = config.get("use_gpu", False)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Process the file
    process_file(
        input_path=input_path,
        output_path=output_path,
        vocab_path=vocab_path,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        mlm_encoded=mlm_encoded,
        device=device  # Pass device to processing function
    )
    
    # Print to stdout (goes to .out file) - keep minimal
    print(f"Birthday token insertion job completed successfully!")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()