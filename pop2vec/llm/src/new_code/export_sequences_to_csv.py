#!/usr/bin/env python3
"""
Script to export life sequences from HDF5 to CSV format with expanded vocabulary.
Useful for inspecting and debugging sequence data.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Import existing dataset class
from pop2vec.llm.src.new_code.load_data import CustomLazyHDF5Dataset
from pop2vec.llm.src.new_code.utils import load_vocab_df, ids_to_tokens

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class SequenceExporter:
    """Handles exporting HDF5 sequences to CSV format"""
    
    def __init__(self, h5_path: str, vocab_path: str):
        """
        Args:
            h5_path: Path to HDF5 sequence file
            vocab_path: Path to vocabulary CSV file
        """
        self.h5_path = h5_path
        self.vocab_path = vocab_path
        
        # Load vocabulary
        self.vocab_df = load_vocab_df(vocab_path)
        
        # Create mappings
        self._create_vocab_mappings()
        
        # Get special token IDs
        self.pad_id = self._get_token_id('[PAD]', 0)
        self.cls_id = self._get_token_id('[CLS]', 1)
        self.sep_id = self._get_token_id('[SEP]', 2)
        
        logger.info(f"Loaded vocabulary with {len(self.vocab_df)} tokens")
        
    def _create_vocab_mappings(self):
        """Create token ID to token/category mappings"""
        # Handle both uppercase and lowercase column names
        if 'TOKEN' in self.vocab_df.columns:
            id_col, token_col, cat_col = 'ID', 'TOKEN', 'CATEGORY'
        else:
            id_col, token_col, cat_col = 'id', 'token', 'category'
            
        self.id_to_token = dict(zip(self.vocab_df[id_col], self.vocab_df[token_col]))
        self.id_to_category = dict(zip(self.vocab_df[id_col], self.vocab_df[cat_col]))
        
    def _get_token_id(self, token_name: str, default: int) -> int:
        """Get token ID by name with fallback"""
        for token_id, token in self.id_to_token.items():
            if token == token_name:
                return token_id
        return default
    
    def export_sequences(
        self,
        sequence_ids: List[int],
        output_dir: str,
        file_prefix: str = "sequence",
        separate_files: bool = False,
        include_padding: bool = False,
        mlm_encoded: bool = False
    ):
        """
        Export sequences to CSV format.
        
        Args:
            sequence_ids: List of sequence IDs to export
            output_dir: Output directory for CSV files
            file_prefix: Prefix for output files
            separate_files: If True, create separate file for each sequence
            include_padding: If True, include padding tokens in output
            mlm_encoded: Whether HDF5 contains MLM data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dataset
        dataset = CustomLazyHDF5Dataset(
            self.h5_path,
            inference=True,
            mlm_encoded=mlm_encoded,
            return_index=True
        )
        
        logger.info(f"Exporting {len(sequence_ids)} sequences from dataset with {len(dataset)} total sequences")
        
        if separate_files:
            self._export_separate_files(dataset, sequence_ids, output_dir, file_prefix, include_padding)
        else:
            self._export_single_file(dataset, sequence_ids, output_dir, file_prefix, include_padding)
    
    def _export_separate_files(
        self, 
        dataset: CustomLazyHDF5Dataset, 
        sequence_ids: List[int], 
        output_dir: str, 
        file_prefix: str,
        include_padding: bool
    ):
        """Export each sequence to a separate CSV file"""
        for seq_idx in tqdm(sequence_ids, desc="Exporting sequences"):
            if seq_idx >= len(dataset):
                logger.warning(f"Sequence index {seq_idx} out of bounds (dataset size: {len(dataset)})")
                continue
                
            # Get sample
            sample = dataset[seq_idx]
            
            # Convert to CSV data
            csv_data = self._sample_to_csv_data(sample, seq_idx, include_padding)
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            output_path = Path(output_dir) / f"{file_prefix}_{seq_idx:06d}.csv"
            df.to_csv(output_path, index=False)
            
        logger.info(f"Exported {len(sequence_ids)} sequences to separate files in {output_dir}")
    
    def _export_single_file(
        self, 
        dataset: CustomLazyHDF5Dataset, 
        sequence_ids: List[int], 
        output_dir: str, 
        file_prefix: str,
        include_padding: bool
    ):
        """Export all sequences to a single CSV file"""
        all_data = []
        
        for seq_idx in tqdm(sequence_ids, desc="Processing sequences"):
            if seq_idx >= len(dataset):
                logger.warning(f"Sequence index {seq_idx} out of bounds (dataset size: {len(dataset)})")
                continue
                
            # Get sample
            sample = dataset[seq_idx]
            
            # Convert to CSV data
            csv_data = self._sample_to_csv_data(sample, seq_idx, include_padding)
            all_data.extend(csv_data)
        
        # Create DataFrame and save
        df = pd.DataFrame(all_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"{file_prefix}_combined_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(sequence_ids)} sequences to {output_path}")
    
    def _sample_to_csv_data(
        self, 
        sample: Dict[str, torch.Tensor], 
        sequence_idx: int, 
        include_padding: bool
    ) -> List[Dict[str, Any]]:
        """Convert a sample to CSV row data"""
        input_ids = sample["input_ids"]  # (4, L)
        padding_mask = sample["padding_mask"]  # (L,)
        
        # Get sequence ID if available
        seq_id = sample.get("sequence_id", sequence_idx)
        if isinstance(seq_id, torch.Tensor):
            seq_id = int(seq_id.item())
        
        # Determine real sequence length
        real_len = int(padding_mask.sum().item()) if include_padding else len(padding_mask)
        if not include_padding:
            # Find actual end of sequence
            real_len = int(padding_mask.sum().item())
        
        csv_rows = []
        
        for pos in range(real_len):
            token_id = int(input_ids[0, pos].item())
            abspos = int(input_ids[1, pos].item())
            age = int(input_ids[2, pos].item())
            segment = int(input_ids[3, pos].item())
            
            # Skip padding tokens unless explicitly requested
            if not include_padding and token_id == self.pad_id:
                continue
            
            # Get token name and category
            token_name = self.id_to_token.get(token_id, f"UNK_{token_id}")
            category = self.id_to_category.get(token_id, "UNKNOWN")
            
            csv_rows.append({
                "sequence_id": seq_id,
                "position": pos,
                "date": abspos,  # Absolute position (days since first event)
                "age": age,
                "segment": segment,
                "token_id": token_id,
                "token": token_name,
                "category": category
            })
        
        return csv_rows
    
    def export_vocabulary_stats(self, output_path: str):
        """Export vocabulary statistics to CSV"""
        # Count token usage if we have category information
        stats_data = []
        
        for _, row in self.vocab_df.iterrows():
            if 'TOKEN' in self.vocab_df.columns:
                token_id = row['ID']
                token_name = row['TOKEN']
                category = row.get('CATEGORY', 'UNKNOWN')
            else:
                token_id = row['id']
                token_name = row['token']
                category = row.get('category', 'UNKNOWN')
            
            stats_data.append({
                'token_id': token_id,
                'token': token_name,
                'category': category,
                'is_special': token_name.startswith('[') and token_name.endswith(']'),
                'is_birthday': token_name.startswith('BIRTHDAY_YEAR_'),
                'is_birth_year': token_name.startswith('BIRTH_YEAR_')
            })
        
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_csv(output_path, index=False)
        logger.info(f"Exported vocabulary statistics to {output_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path) as f:
        return json.load(f)


def parse_sequence_ids(ids_input: str) -> List[int]:
    """Parse sequence IDs from various input formats"""
    if os.path.isfile(ids_input):
        # Read from file (one ID per line)
        with open(ids_input) as f:
            return [int(line.strip()) for line in f if line.strip().isdigit()]
    else:
        # Parse from string (comma-separated or range)
        if ',' in ids_input:
            return [int(x.strip()) for x in ids_input.split(',')]
        elif '-' in ids_input:
            start, end = map(int, ids_input.split('-'))
            return list(range(start, end + 1))
        else:
            return [int(ids_input)]


def main():
    parser = argparse.ArgumentParser(
        description="Export life sequences from HDF5 to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export specific sequences to separate files
  python export_sequences_to_csv.py --h5-path data.h5 --vocab-path vocab.csv --sequence-ids "0,1,2,5" --separate-files

  # Export range of sequences to single file
  python export_sequences_to_csv.py --h5-path data.h5 --vocab-path vocab.csv --sequence-ids "0-10" --output-dir exports

  # Export from file list
  python export_sequences_to_csv.py --h5-path data.h5 --vocab-path vocab.csv --sequence-ids sequence_list.txt
        """
    )
    
    parser.add_argument("--h5-path", required=True, help="Path to HDF5 sequence file")
    parser.add_argument("--vocab-path", required=True, help="Path to vocabulary CSV file")
    parser.add_argument("--sequence-ids", required=True, 
                       help="Sequence IDs to export (comma-separated, range with dash, or file path)")
    parser.add_argument("--output-dir", default="./sequence_exports", help="Output directory")
    parser.add_argument("--file-prefix", default="sequence", help="Prefix for output files")
    parser.add_argument("--separate-files", action="store_true", 
                       help="Create separate file for each sequence")
    parser.add_argument("--include-padding", action="store_true", 
                       help="Include padding tokens in output")
    parser.add_argument("--mlm-encoded", action="store_true", 
                       help="HDF5 contains MLM-encoded data")
    parser.add_argument("--export-vocab-stats", action="store_true",
                       help="Also export vocabulary statistics")
    
    args = parser.parse_args()
    
    # Parse sequence IDs
    try:
        sequence_ids = parse_sequence_ids(args.sequence_ids)
        logger.info(f"Will export {len(sequence_ids)} sequences: {sequence_ids[:10]}{'...' if len(sequence_ids) > 10 else ''}")
    except Exception as e:
        logger.error(f"Failed to parse sequence IDs: {e}")
        return 1
    
    # Create exporter
    try:
        exporter = SequenceExporter(args.h5_path, args.vocab_path)
    except Exception as e:
        logger.error(f"Failed to create exporter: {e}")
        return 1
    
    # Export sequences
    try:
        exporter.export_sequences(
            sequence_ids=sequence_ids,
            output_dir=args.output_dir,
            file_prefix=args.file_prefix,
            separate_files=args.separate_files,
            include_padding=args.include_padding,
            mlm_encoded=args.mlm_encoded
        )
    except Exception as e:
        logger.error(f"Failed to export sequences: {e}")
        return 1
    
    # Export vocabulary stats if requested
    if args.export_vocab_stats:
        try:
            vocab_stats_path = Path(args.output_dir) / "vocabulary_stats.csv"
            exporter.export_vocabulary_stats(str(vocab_stats_path))
        except Exception as e:
            logger.error(f"Failed to export vocabulary stats: {e}")
            return 1
    
    logger.info("Export completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
