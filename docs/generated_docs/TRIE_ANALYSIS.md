# Sequence Trie Analysis

Tools for building and visualizing sequence tries from life sequence data to understand common patterns and token co-occurrences.

## Overview

The trie (prefix tree) structure efficiently represents common sequence patterns by:
- Sharing prefixes among similar sequences
- Counting occurrence frequencies
- Identifying common paths and branches
- Supporting efficient pattern analysis

## Files

### Core Scripts

1. **`build_sequence_trie.py`** - Build trie from HDF5 data
   - Processes sequences and builds trie structure
   - Prunes low-frequency nodes
   - Exports to CSV format

2. **`visualize_trie.py`** - Generate interactive visualization
   - Creates D3.js-based HTML visualization
   - Supports multiple layout modes (tree, cluster, radial)
   - Interactive filtering and zooming

### Configuration

- **`configs/Snellius/build_trie_config.json`** - Configuration for trie building
- **`slurm_scripts/snellius/build_trie.sh`** - SLURM job script

## Usage

### 1. Build Trie from HDF5 Data

#### Using SLURM (Recommended)

```bash
# Edit config if needed
vim pop2vec/llm/configs/Snellius/build_trie_config.json

# Submit job
sbatch pop2vec/llm/slurm_scripts/snellius/build_trie.sh
```

#### Using Python Directly

```bash
python -m pop2vec.llm.src.new_code.build_sequence_trie \
    pop2vec/llm/configs/Snellius/build_trie_config.json
```

### 2. Visualize Trie

```bash
python -m pop2vec.llm.src.new_code.visualize_trie \
    /projects/0/prjs1589/stonybrook/fake_data_v0/step5/trie_analysis/dryrun_trie.csv \
    /projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv \
    --output trie_visualization.html \
    --title "Life Sequence Patterns"
```

Then open `trie_visualization.html` in a web browser.

## Configuration Parameters

### Trie Building (`build_trie_config.json`)

```json
{
    "input_file": "path/to/input.h5",
    "output_file": "path/to/output/trie.csv",
    "vocab_file": "path/to/vocab.csv",
    "lower_limit": 10,          // Min count to keep a node
    "max_nodes": 100000,        // Max nodes in final trie
    "max_sequences": null,      // Max sequences to process (null = all)
    "skip_background": true,    // Skip demographic tokens before first SEP
    "max_seq_len": 512,         // Max sequence length to process
    "mlm_encoded": false        // Whether HDF5 is MLM-encoded
}
```

### Key Parameters

**`lower_limit`** (default: 10)
- Minimum count for a node to be kept in the final trie
- Higher values = more pruning, smaller trie
- Lower values = keep rare patterns, larger trie

**`max_nodes`** (default: 100000)
- Maximum number of nodes in the final trie
- If exceeded, keeps top-N by count
- Automatically includes all ancestors to maintain connectivity

**`skip_background`** (default: true)
- If true, skips demographic tokens before first SEP
- Focuses analysis on life events rather than demographics
- Set to false to include full sequences

**`max_sequences`** (default: null)
- Limit number of sequences processed
- Useful for quick tests or sampling
- null = process all sequences

## Output Files

### Trie CSV Format

The output CSV contains one row per node with these columns:

| Column | Description |
|--------|-------------|
| `node_id` | Unique node identifier (0 = root) |
| `token` | Token ID from vocabulary |
| `parent` | Parent node ID (-1 for root) |
| `count` | Number of sequences passing through this node |
| `end_count` | Number of sequences ending at this node |
| `child_list` | JSON dictionary: {token_id: child_node_id} |

### Metadata JSON

A companion `*_metadata.json` file contains:
- Configuration used
- Special token IDs (CLS, SEP, PAD)
- Statistics before and after pruning
- Input/output file paths

## Examples

### Quick Test (1000 sequences)

```json
{
    "input_file": "/path/to/data.h5",
    "output_file": "/path/to/trie_test.csv",
    "vocab_file": "/path/to/vocab.csv",
    "lower_limit": 5,
    "max_nodes": 10000,
    "max_sequences": 1000,
    "skip_background": true
}
```

### Full Analysis

```json
{
    "input_file": "/path/to/data.h5",
    "output_file": "/path/to/trie_full.csv",
    "vocab_file": "/path/to/vocab.csv",
    "lower_limit": 50,
    "max_nodes": 500000,
    "max_sequences": null,
    "skip_background": true
}
```

### Include Demographics

```json
{
    "input_file": "/path/to/data.h5",
    "output_file": "/path/to/trie_with_bg.csv",
    "vocab_file": "/path/to/vocab.csv",
    "lower_limit": 10,
    "max_nodes": 100000,
    "max_sequences": null,
    "skip_background": false
}
```

## Visualization Features

The interactive HTML visualization includes:

### Controls
- **Min Count**: Filter nodes by minimum occurrence count
- **Max Depth**: Limit tree depth for clarity
- **Layout**: Choose between tree, cluster, or radial layouts
- **Zoom/Pan**: Interactive navigation

### Visual Encoding
- **Node Size**: Proportional to `sqrt(count)`
- **Node Color**: Intensity based on `log(count)`
- **Hover**: Shows node details (name, count, end_count, depth)

### Statistics Panel
- Total nodes in view
- Maximum depth
- Total sequences represented

## Analysis Use Cases

### 1. Common Sequence Patterns
Identify the most common life event sequences:
```bash
# Build with moderate pruning
lower_limit: 100
max_nodes: 50000
skip_background: true
```

### 2. Rare Event Analysis
Keep rare patterns for anomaly detection:
```bash
# Build with minimal pruning
lower_limit: 5
max_nodes: 500000
skip_background: true
```

### 3. Demographic Patterns
Analyze how demographics affect sequences:
```bash
# Include background tokens
skip_background: false
lower_limit: 50
```

### 4. Quick Exploration
Fast iteration for data exploration:
```bash
# Sample 10k sequences
max_sequences: 10000
lower_limit: 10
max_nodes: 10000
```

## Performance Tips

### Memory Usage
- Each node: ~100 bytes
- 100k nodes: ~10 MB
- 1M nodes: ~100 MB
- Pruning reduces memory significantly

### Processing Speed
- ~1000-5000 sequences/second (CPU)
- Depends on sequence length and complexity
- Use `max_sequences` for testing

### Visualization
- For large tries (>10k nodes), use filters:
  - Increase `min_count`
  - Decrease `max_depth`
- Radial layout works well for balanced trees
- Tree layout works well for deep, narrow trees

## Troubleshooting

### Issue: Out of memory
**Solution**: Increase `lower_limit` or decrease `max_nodes`

### Issue: Visualization too cluttered
**Solution**: Use filters in the HTML controls or increase `min_count` when generating

### Issue: Missing expected patterns
**Solution**: Decrease `lower_limit` or check `skip_background` setting

### Issue: Trie disconnected/invalid
**Solution**: The algorithm automatically includes ancestors, but check that `lower_limit` isn't too high

## Advanced Usage

### Reconstructing Trie from CSV

```python
import pandas as pd
import json

df = pd.read_csv("trie.csv")

# Build node dictionary
nodes = {}
for _, row in df.iterrows():
    nodes[row['node_id']] = {
        'token': row['token'],
        'parent': row['parent'],
        'count': row['count'],
        'end_count': row['end_count'],
        'children': json.loads(row['child_list'])
    }

# Navigate tree
root = nodes[0]
print(f"Root: {root}")
```

### Finding Most Common Paths

```python
def find_top_paths(df, vocab_df, n=10):
    """Find top N most common complete paths"""
    # Get nodes with high end_count
    end_nodes = df[df['end_count'] > 0].nlargest(n, 'end_count')
    
    paths = []
    for _, node in end_nodes.iterrows():
        # Trace back to root
        path = []
        current = node
        while current['parent'] != -1:
            token_name = vocab_df[vocab_df['ID'] == current['token']]['TOKEN'].values[0]
            path.append(token_name)
            current = df[df['node_id'] == current['parent']].iloc[0]
        
        paths.append({
            'path': ' -> '.join(reversed(path)),
            'count': node['end_count']
        })
    
    return paths
```

## Related Files

- Input data: `/projects/0/prjs1589/stonybrook/fake_data_v0/step5/encoding=nomlm/dryrun_encoded.h5`
- Vocabulary: `/projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv`
- Output directory: `/projects/0/prjs1589/stonybrook/fake_data_v0/step5/trie_analysis/`
- Logs: `/projects/0/prjs1589/stonybrook/logs/build_trie-*.{out,err}`

## Future Enhancements

Potential additions:
1. Path frequency analysis
2. Subsequence mining
3. Transition probability matrices
4. Anomaly detection
5. Pattern clustering
6. Time-based filtering
7. Multi-file comparison
