#!/usr/bin/env python3
"""
Repair script for malformed generative inference output files.

This script fixes two types of issues:
1. Long sequences (horizon > 20 or prefix_len > 20) that are split across multiple lines
2. Duplicate entries for the same sequence due to parallel GPU execution

Usage:
    python repair_generative_output.py <input_file> <output_file>
    
Example:
    python repair_generative_output.py \
        /projects/0/prjs1589/stonybrook/llm/gen_out/pretty_tokens_20251212.txt \
        /projects/0/prjs1589/stonybrook/llm/gen_out/pretty_tokens_20251212_repaired.txt
"""

import argparse
import re
from collections import defaultdict
from typing import List, Tuple, Dict


def parse_line(line: str) -> Tuple[str, int, int, List[str]]:
    """
    Parse a line from the output file.
    
    Returns:
        (line_type, sequence_num, token_count, tokens)
        where line_type is 'ORIGINAL' or 'GENERATED'
    """
    parts = line.strip().split(',', 2)  # Split into: type+seq, count, tokens
    if len(parts) < 3:
        return None, None, None, None
    
    # Extract type and sequence number
    # E.g., "ORIGINAL PREFIX TOKENS (Sequence 1)" or "GENERATED TOKENS (Sequence 1)"
    match = re.match(r'(ORIGINAL|GENERATED).*\(Sequence (\d+)\)', parts[0])
    if not match:
        return None, None, None, None
    
    line_type = match.group(1)
    seq_num = int(match.group(2))
    token_count = int(parts[1])
    tokens = parts[2].split(',') if len(parts) > 2 else []
    
    return line_type, seq_num, token_count, tokens


def merge_continuation_lines(lines: List[str]) -> List[str]:
    """
    Merge lines that are continuations of the same sequence.
    
    When max_per_line=20 was used, long sequences are split across multiple lines.
    This function merges them back together.
    """
    merged = []
    current_entry = None
    current_tokens = []
    expected_count = 0
    
    for line in lines:
        line_type, seq_num, token_count, tokens = parse_line(line)
        
        if line_type is None:
            continue
        
        # Create entry identifier
        entry_id = f"{line_type}_SEQ{seq_num}"
        
        if current_entry == entry_id:
            # Continuation of previous line
            current_tokens.extend(tokens)
        else:
            # Save previous entry if it exists
            if current_entry is not None:
                # Reconstruct the line
                if current_entry.startswith("ORIGINAL"):
                    prefix = f"ORIGINAL PREFIX TOKENS (Sequence {current_entry.split('SEQ')[1]})"
                else:
                    prefix = f"GENERATED TOKENS (Sequence {current_entry.split('SEQ')[1]})"
                merged.append(f"{prefix},{expected_count},{','.join(current_tokens)}")
            
            # Start new entry
            current_entry = entry_id
            current_tokens = tokens
            expected_count = token_count
    
    # Don't forget the last entry
    if current_entry is not None:
        if current_entry.startswith("ORIGINAL"):
            prefix = f"ORIGINAL PREFIX TOKENS (Sequence {current_entry.split('SEQ')[1]})"
        else:
            prefix = f"GENERATED TOKENS (Sequence {current_entry.split('SEQ')[1]})"
        merged.append(f"{prefix},{expected_count},{','.join(current_tokens)}")
    
    return merged


def deduplicate_sequences(lines: List[str]) -> List[str]:
    """
    Remove duplicate entries from parallel GPU execution.
    
    When using multiple GPUs, sometimes the same sequence is generated twice.
    Keep only the first occurrence.
    """
    seen = set()
    deduped = []
    
    for line in lines:
        line_type, seq_num, token_count, tokens = parse_line(line)
        if line_type is None:
            continue
        
        entry_id = f"{line_type}_SEQ{seq_num}"
        if entry_id not in seen:
            seen.add(entry_id)
            deduped.append(line)
    
    return deduped


def ensure_alternating_pattern(lines: List[str]) -> List[str]:
    """
    Ensure the output follows the pattern: ORIGINAL, GENERATED, ORIGINAL, GENERATED, ...
    """
    result = []
    sequence_pairs = defaultdict(lambda: {'ORIGINAL': None, 'GENERATED': None})
    
    # Group by sequence number
    for line in lines:
        line_type, seq_num, token_count, tokens = parse_line(line)
        if line_type is None:
            continue
        
        sequence_pairs[seq_num][line_type] = line
    
    # Output in order
    for seq_num in sorted(sequence_pairs.keys()):
        pair = sequence_pairs[seq_num]
        if pair['ORIGINAL'] is not None:
            result.append(pair['ORIGINAL'])
        if pair['GENERATED'] is not None:
            result.append(pair['GENERATED'])
    
    return result


def repair_file(input_file: str, output_file: str):
    """
    Repair a malformed generative inference output file.
    """
    print(f"Reading from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Original line count: {len(lines)}")
    
    # Step 1: Merge continuation lines
    print("Step 1: Merging continuation lines...")
    merged = merge_continuation_lines(lines)
    print(f"After merging: {len(merged)} lines")
    
    # Step 2: Deduplicate
    print("Step 2: Deduplicating...")
    deduped = deduplicate_sequences(merged)
    print(f"After deduplication: {len(deduped)} lines")
    
    # Step 3: Ensure alternating pattern
    print("Step 3: Ensuring alternating pattern...")
    final = ensure_alternating_pattern(deduped)
    print(f"Final line count: {len(final)} lines")
    
    # Write output
    print(f"Writing to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in final:
            f.write(line + '\n')
    
    print("Done!")
    
    # Summary
    num_sequences = len(final) // 2
    print(f"\nSummary:")
    print(f"  - Total sequences: {num_sequences}")
    print(f"  - Lines removed: {len(lines) - len(final)}")


def main():
    parser = argparse.ArgumentParser(
        description="Repair malformed generative inference output files"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input file (potentially malformed)"
    )
    parser.add_argument(
        "output_file",
        help="Path to write the repaired output"
    )
    
    args = parser.parse_args()
    repair_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
