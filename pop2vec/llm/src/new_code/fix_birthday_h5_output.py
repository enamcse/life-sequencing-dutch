#!/usr/bin/env python3
"""
Script to fix the HDF5 output from birthday token insertion.

This script:
1. Copies input_ids[0] (token stream) to original_sequence
2. Recalculates padding_mask based on PAD tokens in input_ids[0]
3. Applies gzip compression to reduce file size

The birthday token insertion script only modified input_ids but forgot to:
- Update original_sequence (should match input_ids[0])
- Recalculate padding_mask (1 for real tokens, 0 for padding)
- Apply compression (causing 15x file size increase)
"""

import argparse
import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def fix_h5_file(
    input_path: str,
    output_path: str,
    pad_id: int = 0,
    batch_size: int = 10000,
    compression: str = "gzip",
    compression_opts: int = 4,
):
    """
    Fix the HDF5 file by:
    1. Copying input_ids[0] to original_sequence
    2. Recalculating padding_mask (1 for real tokens, 0 for PAD)
    3. Applying compression
    
    Args:
        input_path: Path to the broken HDF5 file (from birthday insertion)
        output_path: Path to write the fixed HDF5 file
        pad_id: Token ID for padding (default: 0)
        batch_size: Number of samples to process at once
        compression: Compression algorithm (default: gzip)
        compression_opts: Compression level 1-9 (default: 4)
    """
    logger.info("=" * 80)
    logger.info("FIXING BIRTHDAY TOKEN HDF5 OUTPUT")
    logger.info("=" * 80)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"PAD ID: {pad_id}")
    logger.info(f"Compression: {compression} (level {compression_opts})")
    logger.info("=" * 80)
    
    # Open input file and get dimensions
    with h5py.File(input_path, 'r') as f_in:
        # Get shapes
        input_ids_shape = f_in['input_ids'].shape  # (N, 4, seq_len)
        n_samples = input_ids_shape[0]
        seq_len = input_ids_shape[2]
        
        logger.info(f"Input file structure:")
        for key in f_in.keys():
            ds = f_in[key]
            logger.info(f"  {key}: shape={ds.shape}, dtype={ds.dtype}, compression={ds.compression}")
        
        logger.info(f"\nTotal samples: {n_samples:,}")
        logger.info(f"Sequence length: {seq_len}")
        
        # Create output file with compression
        os.makedirs(Path(output_path).parent, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f_out:
            # Create datasets with compression
            # Use chunks for efficient I/O
            chunk_size = min(1000, n_samples)
            
            # input_ids: (N, 4, seq_len)
            f_out.create_dataset(
                'input_ids',
                shape=input_ids_shape,
                dtype=np.int64,
                compression=compression,
                compression_opts=compression_opts,
                chunks=(chunk_size, 4, seq_len)
            )
            
            # original_sequence: (N, seq_len) - copy of input_ids[:, 0, :]
            f_out.create_dataset(
                'original_sequence',
                shape=(n_samples, seq_len),
                dtype=np.int64,
                compression=compression,
                compression_opts=compression_opts,
                chunks=(chunk_size, seq_len)
            )
            
            # padding_mask: (N, seq_len) - 1 for real tokens, 0 for padding
            f_out.create_dataset(
                'padding_mask',
                shape=(n_samples, seq_len),
                dtype=np.int64,
                compression=compression,
                compression_opts=compression_opts,
                chunks=(chunk_size, seq_len)
            )
            
            # sequence_id: (N,)
            if 'sequence_id' in f_in:
                f_out.create_dataset(
                    'sequence_id',
                    shape=(n_samples,),
                    dtype=np.int64,
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=(chunk_size,)
                )
            
            # Process in batches
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            logger.info(f"\nProcessing {n_batches} batches of size {batch_size}...")
            
            total_real_tokens = 0
            total_pad_tokens = 0
            
            for batch_idx in tqdm(range(n_batches), desc="Fixing batches"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                # Read input_ids batch
                input_ids_batch = f_in['input_ids'][start_idx:end_idx]  # (batch, 4, seq_len)
                
                # Extract token stream (dimension 0)
                token_stream = input_ids_batch[:, 0, :]  # (batch, seq_len)
                
                # Calculate padding mask: 1 for real tokens, 0 for padding
                # Real token = not PAD (token_id != pad_id)
                padding_mask_batch = (token_stream != pad_id).astype(np.int64)
                
                # Track statistics
                total_real_tokens += padding_mask_batch.sum()
                total_pad_tokens += (padding_mask_batch == 0).sum()
                
                # Write to output
                f_out['input_ids'][start_idx:end_idx] = input_ids_batch
                f_out['original_sequence'][start_idx:end_idx] = token_stream
                f_out['padding_mask'][start_idx:end_idx] = padding_mask_batch
                
                # Copy sequence_id if exists
                if 'sequence_id' in f_in:
                    f_out['sequence_id'][start_idx:end_idx] = f_in['sequence_id'][start_idx:end_idx]
    
    # Get file sizes
    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    
    # Print summary
    logger.info("=" * 80)
    logger.info("FIX COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total samples processed: {n_samples:,}")
    logger.info(f"")
    logger.info("TOKEN STATISTICS:")
    logger.info(f"  Total real tokens:    {total_real_tokens:,}")
    logger.info(f"  Total padding tokens: {total_pad_tokens:,}")
    logger.info(f"  Average seq length:   {total_real_tokens / n_samples:.1f} tokens")
    logger.info(f"")
    logger.info("FILE SIZE:")
    logger.info(f"  Input (uncompressed):  {input_size / (1024**3):.4f} GB")
    logger.info(f"  Output (compressed):   {output_size / (1024**3):.4f} GB")
    logger.info(f"  Compression ratio:     {input_size / output_size:.1f}x")
    logger.info("=" * 80)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Fix HDF5 output from birthday token insertion"
    )
    parser.add_argument("config", help="JSON config file path")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = json.load(f)
    
    # Required config fields
    input_path = config["input_file"]   # The broken HDF5 from birthday insertion
    output_path = config["output_file"] # Where to write fixed HDF5
    
    # Optional config fields
    pad_id = config.get("pad_id", 0)
    batch_size = config.get("batch_size", 10000)
    compression = config.get("compression", "gzip")
    compression_opts = config.get("compression_opts", 4)
    
    # Run fix
    fix_h5_file(
        input_path=input_path,
        output_path=output_path,
        pad_id=pad_id,
        batch_size=batch_size,
        compression=compression,
        compression_opts=compression_opts,
    )
    
    print(f"Fixed HDF5 saved to: {output_path}")


if __name__ == "__main__":
    main()
