#!/usr/bin/env python3
"""
Utility script for converting between token IDs and token strings.

This script provides functions to:
1. Convert token ID sequences to human-readable token strings
2. Convert token strings back to token IDs
3. Export sequences from HDF5 to readable text format
4. Import text sequences back to HDF5 format

Usage:
    # Convert HDF5 sequences to readable format
    python token_id_mapper.py ids_to_tokens \
        --h5_file /path/to/sequences.h5 \
        --vocab_path /path/to/vocab.csv \
        --output /path/to/output.txt \
        --with_category

    # Convert token strings to IDs
    python token_id_mapper.py tokens_to_ids \
        --input /path/to/tokens.txt \
        --vocab_path /path/to/vocab.csv \
        --output /path/to/output.h5

    # Just show mappings for specific IDs
    python token_id_mapper.py show \
        --vocab_path /path/to/vocab.csv \
        --ids 0 1 2 3 100 500
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import json


class TokenIDMapper:
    """Bidirectional mapper between token IDs and token strings."""
    
    def __init__(self, vocab_path: str):
        """Initialize with vocabulary CSV."""
        self.vocab_df = pd.read_csv(vocab_path)
        
        # Create bidirectional mappings
        self.id_to_token = dict(zip(self.vocab_df['ID'], self.vocab_df['TOKEN']))
        self.id_to_category = dict(zip(self.vocab_df['ID'], self.vocab_df['CATEGORY']))
        self.token_to_id = dict(zip(self.vocab_df['TOKEN'], self.vocab_df['ID']))
        
        # For token|category format
        self.vocab_df['token_cat'] = self.vocab_df['TOKEN'] + '|' + self.vocab_df['CATEGORY']
        self.token_cat_to_id = dict(zip(self.vocab_df['token_cat'], self.vocab_df['ID']))
        
        print(f"Loaded vocabulary: {len(self.id_to_token)} tokens")
    
    def ids_to_tokens(
        self, 
        id_list: List[int], 
        with_category: bool = False,
        remove_padding: bool = True,
        pad_id: int = 0
    ) -> List[str]:
        """
        Convert list of token IDs to token strings.
        
        Args:
            id_list: List of token IDs
            with_category: If True, return "token|category" format
            remove_padding: If True, remove padding tokens (default: token ID 0)
            pad_id: The token ID used for padding
            
        Returns:
            List of token strings
        """
        if remove_padding:
            id_list = [tid for tid in id_list if tid != pad_id]
        
        tokens = []
        for tid in id_list:
            if tid in self.id_to_token:
                token = self.id_to_token[tid]
                if with_category:
                    category = self.id_to_category[tid]
                    tokens.append(f"{token}|{category}")
                else:
                    tokens.append(token)
            else:
                tokens.append(f"<UNK:{tid}>")
        
        return tokens
    
    def tokens_to_ids(
        self, 
        token_list: List[str],
        auto_detect_category: bool = True
    ) -> List[int]:
        """
        Convert list of token strings to token IDs.
        
        Args:
            token_list: List of token strings (can be "token" or "token|category" format)
            auto_detect_category: If True, try both formats
            
        Returns:
            List of token IDs
        """
        ids = []
        for token_str in token_list:
            # Try with category first if it contains |
            if '|' in token_str and token_str in self.token_cat_to_id:
                ids.append(self.token_cat_to_id[token_str])
            # Try without category
            elif token_str in self.token_to_id:
                ids.append(self.token_to_id[token_str])
            # Try extracting just the token part if it has |
            elif auto_detect_category and '|' in token_str:
                token_only = token_str.split('|')[0]
                if token_only in self.token_to_id:
                    ids.append(self.token_to_id[token_only])
                else:
                    print(f"Warning: Unknown token: {token_str}")
                    ids.append(-1)  # Unknown token marker
            else:
                print(f"Warning: Unknown token: {token_str}")
                ids.append(-1)  # Unknown token marker
        
        return ids
    
    def id_to_info(self, token_id: int) -> Dict:
        """Get full information about a token ID."""
        if token_id in self.id_to_token:
            return {
                'id': token_id,
                'token': self.id_to_token[token_id],
                'category': self.id_to_category[token_id],
                'token_category': f"{self.id_to_token[token_id]}|{self.id_to_category[token_id]}"
            }
        else:
            return {
                'id': token_id,
                'token': '<UNKNOWN>',
                'category': '<UNKNOWN>',
                'token_category': f'<UNKNOWN:{token_id}>'
            }
    
    def export_h5_to_text(
        self,
        h5_path: str,
        output_path: str,
        with_category: bool = True,
        max_sequences: Optional[int] = None,
        format: str = 'pretty'  # 'pretty', 'csv', 'json'
    ):
        """
        Export sequences from HDF5 to readable text format.
        
        Args:
            h5_path: Path to input HDF5 file
            output_path: Path to output text file
            with_category: Include category in output
            max_sequences: Limit number of sequences to export
            format: Output format ('pretty', 'csv', 'json')
        """
        print(f"Loading sequences from: {h5_path}")
        
        with h5py.File(h5_path, 'r') as f:
            data = f['data']
            n = min(len(data), max_sequences) if max_sequences else len(data)
            
            print(f"Exporting {n} sequences to: {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as out_f:
                for i in range(n):
                    seq = data[i][0]  # Assuming shape (4, L), take first row (token IDs)
                    seq = [int(x) for x in seq if x != 0]  # Remove padding
                    
                    tokens = self.ids_to_tokens(seq, with_category=with_category)
                    
                    if format == 'pretty':
                        out_f.write(f"Sequence {i+1} ({len(tokens)} tokens):\n")
                        out_f.write("  " + ", ".join(tokens) + "\n\n")
                    elif format == 'csv':
                        out_f.write(f"{i+1},{len(tokens)},{','.join(tokens)}\n")
                    elif format == 'json':
                        out_f.write(json.dumps({
                            'sequence_id': i+1,
                            'length': len(tokens),
                            'tokens': tokens
                        }) + "\n")
                    
                    if (i + 1) % 100 == 0:
                        print(f"  Processed {i+1}/{n} sequences...")
        
        print(f"Export complete!")
    
    def import_text_to_h5(
        self,
        input_path: str,
        output_h5_path: str,
        max_length: int = 512,
        pad_id: int = 0
    ):
        """
        Import text sequences to HDF5 format.
        
        Args:
            input_path: Path to input text file (CSV format)
            output_h5_path: Path to output HDF5 file
            max_length: Maximum sequence length (will pad/truncate)
            pad_id: Token ID to use for padding
        """
        print(f"Loading sequences from: {input_path}")
        
        sequences = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                
                # Skip sequence number and length, get tokens
                tokens = parts[2:]
                ids = self.tokens_to_ids(tokens)
                
                # Pad or truncate
                if len(ids) > max_length:
                    ids = ids[:max_length]
                else:
                    ids = ids + [pad_id] * (max_length - len(ids))
                
                sequences.append(ids)
                
                if line_num % 100 == 0:
                    print(f"  Processed {line_num} sequences...")
        
        print(f"Converting {len(sequences)} sequences to HDF5...")
        sequences_array = np.array(sequences, dtype=np.int32)
        
        # Create HDF5 file
        with h5py.File(output_h5_path, 'w') as f:
            # Store as (num_sequences, 4, max_length) to match expected format
            # Fill all 4 streams with same tokens for now (adjust as needed)
            data = np.zeros((len(sequences), 4, max_length), dtype=np.int32)
            data[:, 0, :] = sequences_array  # Token IDs in first stream
            
            f.create_dataset('data', data=data, compression='gzip')
        
        print(f"Saved to: {output_h5_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert between token IDs and token strings"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # ids_to_tokens command
    ids2tokens_parser = subparsers.add_parser(
        'ids_to_tokens',
        help='Convert HDF5 sequences to readable text'
    )
    ids2tokens_parser.add_argument(
        '--h5_file',
        required=True,
        help='Input HDF5 file with sequences'
    )
    ids2tokens_parser.add_argument(
        '--vocab_path',
        required=True,
        help='Path to vocabulary CSV'
    )
    ids2tokens_parser.add_argument(
        '--output',
        required=True,
        help='Output text file'
    )
    ids2tokens_parser.add_argument(
        '--with_category',
        action='store_true',
        help='Include category in output (token|category format)'
    )
    ids2tokens_parser.add_argument(
        '--format',
        choices=['pretty', 'csv', 'json'],
        default='pretty',
        help='Output format'
    )
    ids2tokens_parser.add_argument(
        '--max_sequences',
        type=int,
        help='Maximum number of sequences to export'
    )
    
    # tokens_to_ids command
    tokens2ids_parser = subparsers.add_parser(
        'tokens_to_ids',
        help='Convert text sequences to HDF5'
    )
    tokens2ids_parser.add_argument(
        '--input',
        required=True,
        help='Input text file (CSV format)'
    )
    tokens2ids_parser.add_argument(
        '--vocab_path',
        required=True,
        help='Path to vocabulary CSV'
    )
    tokens2ids_parser.add_argument(
        '--output',
        required=True,
        help='Output HDF5 file'
    )
    tokens2ids_parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length (will pad/truncate)'
    )
    
    # show command
    show_parser = subparsers.add_parser(
        'show',
        help='Show information about specific token IDs'
    )
    show_parser.add_argument(
        '--vocab_path',
        required=True,
        help='Path to vocabulary CSV'
    )
    show_parser.add_argument(
        '--ids',
        nargs='+',
        type=int,
        help='Token IDs to show'
    )
    show_parser.add_argument(
        '--tokens',
        nargs='+',
        help='Token strings to show'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize mapper
    mapper = TokenIDMapper(args.vocab_path)
    
    if args.command == 'ids_to_tokens':
        mapper.export_h5_to_text(
            h5_path=args.h5_file,
            output_path=args.output,
            with_category=args.with_category,
            max_sequences=args.max_sequences,
            format=args.format
        )
    
    elif args.command == 'tokens_to_ids':
        mapper.import_text_to_h5(
            input_path=args.input,
            output_h5_path=args.output,
            max_length=args.max_length
        )
    
    elif args.command == 'show':
        if args.ids:
            print("\n" + "="*60)
            print("TOKEN ID INFORMATION")
            print("="*60)
            for tid in args.ids:
                info = mapper.id_to_info(tid)
                print(f"\nID: {info['id']}")
                print(f"  Token: {info['token']}")
                print(f"  Category: {info['category']}")
                print(f"  Full: {info['token_category']}")
        
        if args.tokens:
            print("\n" + "="*60)
            print("TOKEN STRING INFORMATION")
            print("="*60)
            for token_str in args.tokens:
                ids = mapper.tokens_to_ids([token_str])
                if ids and ids[0] != -1:
                    info = mapper.id_to_info(ids[0])
                    print(f"\nToken: {token_str}")
                    print(f"  ID: {info['id']}")
                    print(f"  Category: {info['category']}")
                else:
                    print(f"\nToken: {token_str}")
                    print(f"  ID: NOT FOUND")


if __name__ == "__main__":
    main()
