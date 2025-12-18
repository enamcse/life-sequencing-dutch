#!/usr/bin/env python3
"""
Build a memory-efficient trie tree from life sequence data.
Outputs a CSV file that can be used to reconstruct the trie and visualize common patterns.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

# Import existing dataset class
from pop2vec.llm.src.new_code.load_data import CustomLazyHDF5Dataset

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class TrieNode:
    """Memory-efficient trie node for sequence analysis"""
    
    def __init__(self, node_id: int, token: int, parent_id: int = -1):
        self.node_id = node_id
        self.token = token
        self.parent_id = parent_id
        self.count = 0  # Number of sequences passing through this node
        self.end_count = 0  # Number of sequences ending at this node
        self.children = {}  # {token_id: child_node_id}
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary for CSV export"""
        return {
            'node_id': self.node_id,
            'token': self.token,
            'parent': self.parent_id,
            'count': self.count,
            'end_count': self.end_count,
            'child_list': json.dumps(self.children) if self.children else '{}'
        }


class SequenceTrie:
    """Memory-efficient trie tree for sequence analysis"""
    
    def __init__(self, cls_token_id: int = 1, sep_token_id: int = 2, pad_token_id: int = 0):
        self.nodes = {}  # {node_id: TrieNode}
        self.next_node_id = 0
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        
        # Create root node (CLS token)
        self.root = TrieNode(self.next_node_id, cls_token_id, parent_id=-1)
        self.nodes[self.root.node_id] = self.root
        self.next_node_id += 1
        
        logger.info(f"Initialized trie with root node (token={cls_token_id})")
    
    def insert_sequence(self, tokens: List[int]) -> None:
        """
        Insert a token sequence into the trie.
        
        Args:
            tokens: List of token IDs (should start with CLS and include SEP)
        """
        if not tokens or len(tokens) < 2:
            return
        
        # Start from root (CLS token should be first)
        current_node = self.root
        current_node.count += 1
        
        # Skip the first token (CLS) since root is CLS
        for i, token in enumerate(tokens[1:], start=1):
            # Stop at padding or if we've processed enough
            if token == self.pad_token_id:
                break
            
            # Check if child exists
            if token in current_node.children:
                # Navigate to existing child
                child_id = current_node.children[token]
                current_node = self.nodes[child_id]
                current_node.count += 1
            else:
                # Create new child node
                new_node = TrieNode(
                    node_id=self.next_node_id,
                    token=token,
                    parent_id=current_node.node_id
                )
                new_node.count = 1
                
                # Link parent to child
                current_node.children[token] = new_node.node_id
                self.nodes[new_node.node_id] = new_node
                self.next_node_id += 1
                
                current_node = new_node
            
            # Check if this is the last non-padding token
            is_last = (i == len(tokens) - 1 or 
                      (i < len(tokens) - 1 and tokens[i + 1] == self.pad_token_id))
            if is_last:
                current_node.end_count += 1
    
    def prune(self, lower_limit: int = 10, max_nodes: int = 100000) -> None:
        """
        Prune the trie to keep only high-frequency paths.
        
        Args:
            lower_limit: Minimum count to keep a node
            max_nodes: Maximum number of nodes to keep
        """
        logger.info(f"Pruning trie: lower_limit={lower_limit}, max_nodes={max_nodes}")
        logger.info(f"Nodes before pruning: {len(self.nodes)}")
        
        # Step 1: Filter by count threshold
        nodes_above_threshold = {
            node_id: node for node_id, node in self.nodes.items()
            if node.count >= lower_limit or node_id == 0  # Keep root
        }
        
        logger.info(f"Nodes above threshold ({lower_limit}): {len(nodes_above_threshold)}")
        
        # Step 2: If still too many, keep top N by count
        if len(nodes_above_threshold) > max_nodes:
            # Sort by count (descending) and keep top max_nodes
            sorted_nodes = sorted(
                nodes_above_threshold.items(),
                key=lambda x: x[1].count,
                reverse=True
            )[:max_nodes]
            
            # Ensure all parents are included to maintain connectivity
            kept_node_ids = set(node_id for node_id, _ in sorted_nodes)
            
            # Add all ancestors of kept nodes (iterative to avoid recursion limit)
            nodes_to_check = list(kept_node_ids)
            while nodes_to_check:
                node_id = nodes_to_check.pop()
                node = nodes_above_threshold.get(node_id)
                if node and node.parent_id != -1 and node.parent_id not in kept_node_ids:
                    kept_node_ids.add(node.parent_id)
                    nodes_to_check.append(node.parent_id)
            
            nodes_above_threshold = {
                node_id: node for node_id, node in nodes_above_threshold.items()
                if node_id in kept_node_ids
            }
            
            logger.info(f"Nodes after max_nodes limit ({max_nodes}): {len(nodes_above_threshold)}")
        
        # Step 3: Update trie with pruned nodes
        old_nodes = self.nodes
        self.nodes = nodes_above_threshold
        
        # Step 4: Clean up children lists (remove references to deleted nodes)
        for node in self.nodes.values():
            node.children = {
                token: child_id for token, child_id in node.children.items()
                if child_id in self.nodes
            }
        
        logger.info(f"Nodes after pruning: {len(self.nodes)}")
        logger.info(f"Reduction: {len(old_nodes) - len(self.nodes)} nodes removed")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trie to a pandas DataFrame for CSV export"""
        rows = [node.to_dict() for node in self.nodes.values()]
        df = pd.DataFrame(rows)
        
        # Sort by node_id for consistency
        df = df.sort_values('node_id').reset_index(drop=True)
        
        return df
    
    def get_statistics(self) -> Dict:
        """Get statistics about the trie"""
        max_depth = 0
        total_end_nodes = 0
        leaf_nodes = 0
        
        # Iterative depth calculation to avoid RecursionError on deep tries
        # Use BFS to find the maximum depth
        if self.root.node_id in self.nodes:
            queue = [(self.root.node_id, 0)]  # (node_id, depth)
            while queue:
                node_id, depth = queue.pop(0)
                max_depth = max(max_depth, depth)
                
                node = self.nodes.get(node_id)
                if node and node.children:
                    for child_id in node.children.values():
                        queue.append((child_id, depth + 1))
        
        for node in self.nodes.values():
            if node.end_count > 0:
                total_end_nodes += 1
            if not node.children:
                leaf_nodes += 1
        
        return {
            'total_nodes': len(self.nodes),
            'max_depth': max_depth,
            'total_sequences': self.root.count,
            'leaf_nodes': leaf_nodes,
            'nodes_with_endings': total_end_nodes,
        }


def build_trie_from_hdf5(
    input_path: str,
    vocab_path: str,
    output_path: str,
    lower_limit: int = 10,
    max_nodes: int = 100000,
    max_sequences: Optional[int] = None,
    skip_background: bool = True,
    max_seq_len: int = 512,
    mlm_encoded: bool = False
) -> None:
    """
    Build a trie tree from HDF5 sequences and export to CSV.
    
    Args:
        input_path: Path to input HDF5 file
        vocab_path: Path to vocabulary CSV file
        output_path: Path to output CSV file
        lower_limit: Minimum count to keep a node
        max_nodes: Maximum number of nodes to keep
        max_sequences: Maximum sequences to process (None = all)
        skip_background: If True, skip background tokens (before first SEP)
        max_seq_len: Maximum sequence length to process
        mlm_encoded: Whether the HDF5 contains MLM data
    """
    # Print to stdout
    print(f"Building trie from: {input_path}")
    print(f"Output: {output_path}")
    
    # Load vocabulary
    logger.info("=" * 80)
    logger.info("BUILDING SEQUENCE TRIE")
    logger.info("=" * 80)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Vocab:  {vocab_path}")
    logger.info("")
    logger.info("CONFIGURATION:")
    logger.info(f"  lower_limit:    {lower_limit}")
    logger.info(f"  max_nodes:      {max_nodes}")
    logger.info(f"  max_sequences:  {max_sequences or 'all'}")
    logger.info(f"  skip_background: {skip_background}")
    logger.info(f"  max_seq_len:    {max_seq_len}")
    logger.info(f"  mlm_encoded:    {mlm_encoded}")
    logger.info("=" * 80)
    
    vocab_df = pd.read_csv(vocab_path)
    
    # Detect column names (handle both upper and lowercase)
    token_col = 'TOKEN' if 'TOKEN' in vocab_df.columns else 'token'
    id_col = 'ID' if 'ID' in vocab_df.columns else 'id'
    
    # Get special token IDs
    token_to_id = dict(zip(vocab_df[token_col], vocab_df[id_col]))
    cls_id = token_to_id.get('[CLS]', 1)
    sep_id = token_to_id.get('[SEP]', 2)
    pad_id = token_to_id.get('[PAD]', 0)
    
    logger.info(f"Special tokens: CLS={cls_id}, SEP={sep_id}, PAD={pad_id}")
    
    # Initialize trie
    trie = SequenceTrie(cls_token_id=cls_id, sep_token_id=sep_id, pad_token_id=pad_id)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = CustomLazyHDF5Dataset(
        input_path,
        inference=True,
        mlm_encoded=mlm_encoded,
        return_index=True
    )
    
    n_samples = len(dataset)
    if max_sequences is not None:
        n_samples = min(n_samples, max_sequences)
    
    logger.info(f"Processing {n_samples:,} sequences...")
    
    # Process sequences
    for i in tqdm(range(n_samples), desc="Building trie"):
        sample = dataset[i]
        input_ids = sample["input_ids"]  # (4, L)
        padding_mask = sample["padding_mask"]  # (L,)
        
        # Get real sequence length
        real_len = int(padding_mask.sum().item())
        
        # Extract token sequence (first dimension only)
        tokens = input_ids[0, :real_len].cpu().numpy().tolist()
        
        # Optionally skip background tokens
        if skip_background and sep_id in tokens:
            # Find first SEP and start from there
            sep_idx = tokens.index(sep_id)
            # Keep CLS at the beginning, then add tokens after SEP
            tokens = [cls_id] + tokens[sep_idx + 1:]
        
        # Truncate if too long
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        
        # Insert into trie
        trie.insert_sequence(tokens)
    
    # Get statistics before pruning
    stats_before = trie.get_statistics()
    logger.info("")
    logger.info("STATISTICS BEFORE PRUNING:")
    logger.info(f"  Total nodes:       {stats_before['total_nodes']:,}")
    logger.info(f"  Max depth:         {stats_before['max_depth']}")
    logger.info(f"  Total sequences:   {stats_before['total_sequences']:,}")
    logger.info(f"  Leaf nodes:        {stats_before['leaf_nodes']:,}")
    logger.info(f"  Nodes with endings: {stats_before['nodes_with_endings']:,}")
    
    # Prune trie
    logger.info("")
    trie.prune(lower_limit=lower_limit, max_nodes=max_nodes)
    
    # Get statistics after pruning
    stats_after = trie.get_statistics()
    logger.info("")
    logger.info("STATISTICS AFTER PRUNING:")
    logger.info(f"  Total nodes:       {stats_after['total_nodes']:,}")
    logger.info(f"  Max depth:         {stats_after['max_depth']}")
    logger.info(f"  Total sequences:   {stats_after['total_sequences']:,}")
    logger.info(f"  Leaf nodes:        {stats_after['leaf_nodes']:,}")
    logger.info(f"  Nodes with endings: {stats_after['nodes_with_endings']:,}")
    
    # Export to CSV
    logger.info("")
    logger.info("Exporting trie to CSV...")
    df = trie.to_dataframe()
    
    # Create output directory if needed
    os.makedirs(Path(output_path).parent, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved trie to: {output_path}")
    logger.info(f"CSV shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Also save metadata JSON
    metadata_path = output_path.replace('.csv', '_metadata.json')
    metadata = {
        'input_file': input_path,
        'vocab_file': vocab_path,
        'config': {
            'lower_limit': lower_limit,
            'max_nodes': max_nodes,
            'max_sequences': max_sequences,
            'skip_background': skip_background,
            'max_seq_len': max_seq_len,
            'mlm_encoded': mlm_encoded
        },
        'special_tokens': {
            'CLS': int(cls_id),
            'SEP': int(sep_id),
            'PAD': int(pad_id)
        },
        'statistics_before_pruning': stats_before,
        'statistics_after_pruning': stats_after
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to: {metadata_path}")
    logger.info("=" * 80)
    logger.info("TRIE BUILDING COMPLETE")
    logger.info("=" * 80)
    
    # Print to stdout
    print(f"Trie building completed successfully!")
    print(f"Output: {output_path}")
    print(f"Nodes: {stats_after['total_nodes']:,}")


def main():
    parser = argparse.ArgumentParser(description="Build sequence trie from life sequence data")
    parser.add_argument("config", help="JSON config file path")
    
    args = parser.parse_args()
    
    # Print to stdout
    print(f"Starting trie building job...")
    print(f"Config: {args.config}")
    
    # Load configuration
    with open(args.config) as f:
        config = json.load(f)
    
    # Required fields
    input_path = config["input_file"]
    output_path = config["output_file"]
    vocab_path = config["vocab_file"]
    
    # Optional fields
    lower_limit = config.get("lower_limit", 10)
    max_nodes = config.get("max_nodes", 100000)
    max_sequences = config.get("max_sequences", None)
    skip_background = config.get("skip_background", True)
    max_seq_len = config.get("max_seq_len", 512)
    mlm_encoded = config.get("mlm_encoded", False)
    
    # Build trie
    build_trie_from_hdf5(
        input_path=input_path,
        vocab_path=vocab_path,
        output_path=output_path,
        lower_limit=lower_limit,
        max_nodes=max_nodes,
        max_sequences=max_sequences,
        skip_background=skip_background,
        max_seq_len=max_seq_len,
        mlm_encoded=mlm_encoded
    )


if __name__ == "__main__":
    main()
