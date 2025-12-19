#!/usr/bin/env python3
"""
Retrofit ground truth continuations into old generative output files.

This script takes an old output file (with only PREFIX and GENERATED) and adds
GROUND TRUTH CONTINUATION by looking up the original sequence in the HDF5 file.

Usage:
    python retrofit_ground_truth.py \
        --input old_output.txt \
        --output new_output_with_ground_truth.txt \
        --h5_file /path/to/encoded.h5 \
        --vocab_path /path/to/vocab.csv \
        [--dataset_key data]

Example:
    python pop2vec/llm/scripts/retrofit_ground_truth.py \
        --input /projects/0/prjs1589/stonybrook/llm/gen_out/pretty_tokens_20251212.txt \
        --output /projects/0/prjs1589/stonybrook/llm/gen_out/pretty_tokens_20251212_gt2.txt \
        --h5_file /projects/0/prjs1589/stonybrook/fake_data_v0/step5/encoding=nomlm/encoded.h5 \
        --vocab_path /projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv
"""

import argparse
import h5py
import pandas as pd
import re
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm


def load_vocab(vocab_path: str) -> pd.DataFrame:
    """Load vocabulary CSV with TOKEN, CATEGORY, ID columns."""
    return pd.read_csv(vocab_path, dtype={"TOKEN": str, "CATEGORY": str, "ID": int})


def ids_to_tokens(id_list: List[int], vocab_df: pd.DataFrame, with_category: bool = True) -> List[str]:
    """Convert token IDs to human-readable strings (TOKEN|CATEGORY format)."""
    tok = vocab_df.set_index("ID")["TOKEN"].to_dict()
    cat = vocab_df.set_index("ID")["CATEGORY"].to_dict()
    
    result = []
    for tid in id_list:
        token = tok.get(int(tid), f"<UNK:{tid}>")
        if with_category:
            category = cat.get(int(tid), "")
            result.append(f"{token}|{category}")
        else:
            result.append(token)
    return result


def tokens_to_ids(token_list: List[str], vocab_df: pd.DataFrame) -> List[int]:
    """Convert token strings (TOKEN|CATEGORY or TOKEN format) to IDs."""
    # Create mappings
    tok_to_id = dict(zip(vocab_df['TOKEN'], vocab_df['ID']))
    
    # Create token|category to ID mapping
    vocab_df['tok_cat'] = vocab_df['TOKEN'] + '|' + vocab_df['CATEGORY']
    tok_cat_to_id = dict(zip(vocab_df['tok_cat'], vocab_df['ID']))
    
    ids = []
    for token_str in token_list:
        # Try with category first
        if '|' in token_str and token_str in tok_cat_to_id:
            ids.append(tok_cat_to_id[token_str])
        # Try without category
        elif token_str in tok_to_id:
            ids.append(tok_to_id[token_str])
        # Try extracting just the token part
        elif '|' in token_str:
            token_only = token_str.split('|')[0]
            if token_only in tok_to_id:
                ids.append(tok_to_id[token_only])
            else:
                raise ValueError(f"Unknown token: {token_str}")
        else:
            raise ValueError(f"Unknown token: {token_str}")
    
    return ids


def parse_old_format_line(line: str) -> Optional[Tuple[str, int, int, List[str]]]:
    """
    Parse a line from old format output.
    
    Returns:
        (line_type, sequence_num, token_count, token_list)
        where line_type is 'PREFIX' or 'GENERATED'
    """
    parts = line.strip().split(',', 2)
    if len(parts) < 3:
        return None
    
    # Match old format: "ORIGINAL PREFIX TOKENS (Sequence N)" or "GENERATED TOKENS (Sequence N)"
    match = re.match(r'(ORIGINAL PREFIX TOKENS|GENERATED TOKENS).*\(Sequence (\d+)\)', parts[0])
    if not match:
        return None
    
    line_type = 'PREFIX' if 'PREFIX' in match.group(1) else 'GENERATED'
    seq_num = int(match.group(2))
    token_count = int(parts[1])
    
    # Parse tokens
    tokens_str = parts[2]
    tokens = [t.strip() for t in tokens_str.split(',') if t.strip()]
    
    return line_type, seq_num, token_count, tokens


def find_sequence_in_h5(
    prefix_ids: List[int],
    h5_file: h5py.File,
    dataset_key: str = 'input_ids',
    max_sequences_to_check: int = 100000
) -> Optional[Tuple[int, List[int]]]:
    """
    Find a sequence in HDF5 that starts with the given prefix.
    
    Returns:
        (sequence_index, full_sequence_ids) or None if not found
    """
    data = h5_file[dataset_key]
    n_sequences = min(len(data), max_sequences_to_check)
    prefix_len = len(prefix_ids)
    
    print(f"  Searching for prefix of length {prefix_len} in {n_sequences} sequences...")
    
    for i in range(n_sequences):
        # Get token sequence (first channel of 4-channel data)
        seq = data[i][0]  # Shape: (512,) assuming (N, 4, 512) format
        
        # Remove padding (assuming 0 is pad)
        seq_no_pad = [int(x) for x in seq if x != 0]
        
        # Check if this sequence starts with our prefix
        if len(seq_no_pad) >= prefix_len:
            if seq_no_pad[:prefix_len] == prefix_ids:
                print(f"  ✓ Found matching sequence at index {i}")
                return i, seq_no_pad
    
    return None


def retrofit_file(
    input_path: str,
    output_path: str,
    h5_path: str,
    vocab_path: str,
    dataset_key: str = 'input_ids',
    max_sequences_to_check: int = 100000
):
    """
    Retrofit ground truth continuations into an old output file.
    """
    print(f"Loading vocabulary from: {vocab_path}")
    vocab_df = load_vocab(vocab_path)
    
    print(f"Opening HDF5 file: {h5_path}")
    h5_file = h5py.File(h5_path, 'r')
    
    print(f"Reading old output file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse all lines
    parsed_data = {}
    for line in lines:
        result = parse_old_format_line(line)
        if result is None:
            continue
        
        line_type, seq_num, token_count, tokens = result
        
        if seq_num not in parsed_data:
            parsed_data[seq_num] = {}
        
        parsed_data[seq_num][line_type] = {
            'count': token_count,
            'tokens': tokens
        }
    
    print(f"Found {len(parsed_data)} sequences in old output")
    
    # Process each sequence
    output_lines = []
    stats = {
        'total': len(parsed_data),
        'found': 0,
        'not_found': 0,
        'no_continuation': 0
    }
    
    for seq_num in tqdm(sorted(parsed_data.keys()), desc="Processing sequences"):
        seq_data = parsed_data[seq_num]
        
        if 'PREFIX' not in seq_data or 'GENERATED' not in seq_data:
            print(f"  ⚠️  Sequence {seq_num} is incomplete, skipping")
            continue
        
        prefix_tokens = seq_data['PREFIX']['tokens']
        generated_tokens = seq_data['GENERATED']['tokens']
        generated_len = len(generated_tokens)
        
        # Convert prefix tokens to IDs
        try:
            prefix_ids = tokens_to_ids(prefix_tokens, vocab_df)
        except ValueError as e:
            print(f"  ⚠️  Sequence {seq_num}: Error converting tokens to IDs: {e}")
            stats['not_found'] += 1
            # Still write PREFIX and GENERATED without ground truth
            output_lines.append(f"ORIGINAL PREFIX TOKENS (Sequence {seq_num}),{len(prefix_tokens)},{','.join(prefix_tokens)}\n")
            output_lines.append(f"GENERATED TOKENS (Sequence {seq_num}),{generated_len},{','.join(generated_tokens)}\n")
            continue
        
        # Find the sequence in HDF5
        result = find_sequence_in_h5(prefix_ids, h5_file, dataset_key, max_sequences_to_check)
        
        if result is None:
            print(f"  ⚠️  Sequence {seq_num}: Could not find matching sequence in HDF5")
            stats['not_found'] += 1
            # Write without ground truth
            output_lines.append(f"ORIGINAL PREFIX TOKENS (Sequence {seq_num}),{len(prefix_tokens)},{','.join(prefix_tokens)}\n")
            output_lines.append(f"GENERATED TOKENS (Sequence {seq_num}),{generated_len},{','.join(generated_tokens)}\n")
            continue
        
        seq_idx, full_sequence_ids = result
        stats['found'] += 1
        
        # Extract ground truth continuation
        prefix_len = len(prefix_ids)
        continuation_end = prefix_len + generated_len
        
        if continuation_end > len(full_sequence_ids):
            print(f"  ⚠️  Sequence {seq_num}: Not enough tokens for continuation (need {continuation_end}, have {len(full_sequence_ids)})")
            stats['no_continuation'] += 1
            ground_truth_ids = full_sequence_ids[prefix_len:]  # Take whatever is available
        else:
            ground_truth_ids = full_sequence_ids[prefix_len:continuation_end]
        
        # Convert ground truth IDs to tokens
        with_category = '|' in prefix_tokens[0] if prefix_tokens else True
        ground_truth_tokens = ids_to_tokens(ground_truth_ids, vocab_df, with_category=with_category)
        
        # Write all three lines
        output_lines.append(f"ORIGINAL PREFIX TOKENS (Sequence {seq_num}),{len(prefix_tokens)},{','.join(prefix_tokens)}\n")
        output_lines.append(f"GROUND TRUTH CONTINUATION (Sequence {seq_num}),{len(ground_truth_tokens)},{','.join(ground_truth_tokens)}\n")
        output_lines.append(f"GENERATED TOKENS (Sequence {seq_num}),{generated_len},{','.join(generated_tokens)}\n")
    
    # Write output file
    print(f"\nWriting retrofitted output to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    # Close HDF5
    h5_file.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("RETROFIT STATISTICS")
    print("="*60)
    print(f"Total sequences: {stats['total']}")
    print(f"Successfully found and added ground truth: {stats['found']}")
    print(f"Could not find in HDF5: {stats['not_found']}")
    print(f"Found but insufficient continuation: {stats['no_continuation']}")
    print(f"\nOutput saved to: {output_path}")
    print("You can now use analyze_generative_output.py on this file!")


def main():
    parser = argparse.ArgumentParser(
        description="Retrofit ground truth continuations into old output files"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to old output file (with PREFIX and GENERATED only)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to new output file (with PREFIX, GROUND TRUTH, GENERATED)"
    )
    parser.add_argument(
        "--h5_file",
        required=True,
        help="Path to HDF5 file with original sequences"
    )
    parser.add_argument(
        "--vocab_path",
        required=True,
        help="Path to vocabulary CSV file"
    )
    parser.add_argument(
        "--dataset_key",
        default="input_ids",
        help="HDF5 dataset key (default: 'input_ids')"
    )
    parser.add_argument(
        "--max_sequences_to_check",
        type=int,
        default=100000,
        help="Maximum number of sequences to check in HDF5 (default: 100000)"
    )
    
    args = parser.parse_args()
    
    retrofit_file(
        input_path=args.input,
        output_path=args.output,
        h5_path=args.h5_file,
        vocab_path=args.vocab_path,
        dataset_key=args.dataset_key,
        max_sequences_to_check=args.max_sequences_to_check
    )


if __name__ == "__main__":
    main()
