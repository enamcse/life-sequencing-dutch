#!/usr/bin/env python3
"""
Quick local test for trie building functionality.
Tests on a small sample without needing SLURM.
"""

import sys
import json
import tempfile
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Now we can import
try:
    from pop2vec.llm.src.new_code.build_sequence_trie import (
        SequenceTrie, 
        build_trie_from_hdf5
    )
except ImportError:
    # If still fails, try direct import from same directory
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_sequence_trie",
        Path(__file__).parent / "build_sequence_trie.py"
    )
    build_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_module)
    SequenceTrie = build_module.SequenceTrie
    build_trie_from_hdf5 = build_module.build_trie_from_hdf5

def test_trie_basic():
    """Test basic trie operations"""
    print("Testing basic trie operations...")
    
    trie = SequenceTrie(cls_token_id=1, sep_token_id=2, pad_token_id=0)
    
    # Insert some test sequences
    sequences = [
        [1, 2, 10, 20, 30, 0, 0],  # CLS, SEP, 10, 20, 30, PAD, PAD
        [1, 2, 10, 20, 40, 0, 0],  # CLS, SEP, 10, 20, 40, PAD, PAD
        [1, 2, 10, 20, 40, 0, 0],  # Duplicate
        [1, 2, 10, 50, 0, 0, 0],   # CLS, SEP, 10, 50, PAD, PAD, PAD
        [1, 2, 15, 25, 0, 0, 0],   # CLS, SEP, 15, 25, PAD, PAD, PAD
    ]
    
    for seq in sequences:
        trie.insert_sequence(seq)
    
    # Check statistics
    stats = trie.get_statistics()
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Total sequences: {stats['total_sequences']}")
    print(f"  Leaf nodes: {stats['leaf_nodes']}")
    
    assert stats['total_sequences'] == 5, "Should have 5 sequences"
    assert stats['max_depth'] > 0, "Should have some depth"
    
    # Test pruning
    print("\nTesting pruning...")
    original_nodes = len(trie.nodes)
    trie.prune(lower_limit=2, max_nodes=100)
    pruned_nodes = len(trie.nodes)
    print(f"  Nodes before pruning: {original_nodes}")
    print(f"  Nodes after pruning: {pruned_nodes}")
    
    # Test DataFrame export
    print("\nTesting DataFrame export...")
    df = trie.to_dataframe()
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    assert 'node_id' in df.columns
    assert 'token' in df.columns
    assert 'count' in df.columns
    assert 'child_list' in df.columns
    
    print("\n✓ Basic trie tests passed!")
    return True


def test_trie_from_config():
    """Test building trie with actual config (small sample)"""
    print("\nTesting trie building from config...")
    
    # Create temporary config
    config = {
        "input_file": "/projects/0/prjs1589/stonybrook/fake_data_v0/step5/encoding=nomlm/dryrun_encoded.h5",
        "output_file": "/projects/0/prjs1589/stonybrook/visualize/trie_tree/test_trie.csv",
        "vocab_file": "/projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv",
        "lower_limit": 5,
        "max_nodes": 1000,
        "max_sequences": 100,  # Only process 100 sequences for quick test
        "skip_background": True,
        "max_seq_len": 512,
        "mlm_encoded": False
    }
    
    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    
    try:
        print(f"  Config: {config_path}")
        print(f"  Processing {config['max_sequences']} sequences...")
        
        # Build trie
        build_trie_from_hdf5(
            input_path=config["input_file"],
            vocab_path=config["vocab_file"],
            output_path=config["output_file"],
            lower_limit=config["lower_limit"],
            max_nodes=config["max_nodes"],
            max_sequences=config["max_sequences"],
            skip_background=config["skip_background"],
            max_seq_len=config["max_seq_len"],
            mlm_encoded=config["mlm_encoded"]
        )
        
        # Check output exists
        import os
        assert os.path.exists(config["output_file"]), "Output CSV should exist"
        assert os.path.exists(config["output_file"].replace('.csv', '_metadata.json')), "Metadata should exist"
        
        # Check CSV content
        import pandas as pd
        df = pd.read_csv(config["output_file"])
        print(f"\n  Output CSV shape: {df.shape}")
        print(f"  Sample rows:")
        print(df.head())
        
        print("\n✓ Config-based trie building test passed!")
        return True
        
    finally:
        # Cleanup
        import os
        if os.path.exists(config_path):
            os.unlink(config_path)


if __name__ == "__main__":
    print("=" * 60)
    print("TRIE BUILDER TEST SUITE")
    print("=" * 60)
    
    try:
        # Run basic tests
        test_trie_basic()
        
        # Run config-based test (requires data files)
        print("\nChecking if data files exist for integration test...")
        import os
        data_exists = os.path.exists("/projects/0/prjs1589/stonybrook/fake_data_v0/step5/encoding=nomlm/dryrun_encoded.h5")
        vocab_exists = os.path.exists("/projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv")
        
        if data_exists and vocab_exists:
            test_trie_from_config()
        else:
            print("  Data files not found, skipping integration test")
            print("  (This is normal if running outside the cluster)")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
