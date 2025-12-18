# Sequence Trie Builder - Quick Reference

## 🚀 Quick Start

### 1. Test Locally (Optional)
```bash
cd ~/life-sequencing-dutch
python pop2vec/llm/src/new_code/test_trie.py
```

### 2. Submit SLURM Job
```bash
sbatch pop2vec/llm/slurm_scripts/snellius/build_trie.sh
```

### 3. Monitor Progress
```bash
# Watch logs
tail -f /projects/0/prjs1589/stonybrook/logs/build_trie-*.err

# Check status
squeue -u $USER
```

### 4. Visualize Results
```bash
python -m pop2vec.llm.src.new_code.visualize_trie \
    /projects/0/prjs1589/stonybrook/fake_data_v0/step5/trie_analysis/dryrun_trie.csv \
    /projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv \
    -o trie_viz.html
```

## 📁 Files Created

```
pop2vec/llm/
├── src/new_code/
│   ├── build_sequence_trie.py      ← Main script
│   ├── visualize_trie.py            ← Visualization
│   └── test_trie.py                 ← Tests
├── configs/Snellius/
│   └── build_trie_config.json       ← Configuration
├── slurm_scripts/snellius/
│   └── build_trie.sh                ← SLURM script
├── TRIE_ANALYSIS.md                 ← Full documentation
└── TRIE_IMPLEMENTATION_SUMMARY.md   ← Implementation details
```

## ⚙️ Configuration

Edit `configs/Snellius/build_trie_config.json`:

```json
{
    "lower_limit": 10,      // Min count to keep
    "max_nodes": 100000,    // Max nodes
    "max_sequences": null,  // null = all
    "skip_background": true // Skip demographics
}
```

## 📊 Output

### CSV Format
`/projects/0/prjs1589/stonybrook/fake_data_v0/step5/trie_analysis/dryrun_trie.csv`

| Column | Description |
|--------|-------------|
| node_id | Unique ID (0 = root) |
| token | Token ID from vocab |
| parent | Parent node ID (-1 = root) |
| count | Times seen |
| end_count | Times ended here |
| child_list | JSON: {token: child_id} |

### Metadata
`dryrun_trie_metadata.json` - Configuration and statistics

## 🎨 Visualization

```bash
# Basic
python -m pop2vec.llm.src.new_code.visualize_trie \
    trie.csv vocab.csv -o viz.html

# With filters
python -m pop2vec.llm.src.new_code.visualize_trie \
    trie.csv vocab.csv -o viz.html \
    --min-count 50 \
    --max-depth 15 \
    --width 1600 \
    --height 1000
```

### Interactive Features
- **Layouts**: Tree, Cluster, Radial
- **Filters**: Min count, max depth
- **Zoom/Pan**: Navigate large trees
- **Hover**: See node details

## 📈 Common Use Cases

### Quick Test (100 sequences)
```json
{
    "max_sequences": 100,
    "lower_limit": 5,
    "max_nodes": 1000
}
```

### Full Analysis
```json
{
    "max_sequences": null,
    "lower_limit": 50,
    "max_nodes": 500000
}
```

### With Demographics
```json
{
    "skip_background": false,
    "lower_limit": 20
}
```

## 🔍 Analysis Examples

### Find Most Common Patterns
```python
import pandas as pd
df = pd.read_csv('trie.csv')

# Top 10 most common paths (by end_count)
top_ends = df.nlargest(10, 'end_count')
print(top_ends[['node_id', 'token', 'count', 'end_count']])

# Top 10 most traversed nodes
top_nodes = df.nlargest(10, 'count')
print(top_nodes[['node_id', 'token', 'count']])
```

### Trace Path Back to Root
```python
def get_path(df, node_id):
    """Get full path from root to node"""
    path = []
    current_id = node_id
    
    while current_id != -1:
        row = df[df['node_id'] == current_id].iloc[0]
        path.append(row['token'])
        current_id = row['parent']
    
    return list(reversed(path))

# Example: trace most common ending
top_end = df.nlargest(1, 'end_count').iloc[0]
path = get_path(df, top_end['node_id'])
print(f"Most common path: {path}")
```

## ⚡ Performance Tips

**Memory**
- 100k nodes ≈ 10 MB
- Increase `lower_limit` if OOM

**Speed**
- ~1000-5000 seq/sec
- Use `max_sequences` for testing

**Visualization**
- Filter large trees (>10k nodes)
- Radial layout for balanced trees
- Tree layout for deep/narrow

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | ↑ `lower_limit` or ↓ `max_nodes` |
| Viz cluttered | Use filters or ↑ `min_count` |
| Missing patterns | ↓ `lower_limit` |
| Slow processing | Use `max_sequences` |

## 📚 Documentation

- **TRIE_ANALYSIS.md**: Full guide (use cases, examples, advanced)
- **TRIE_IMPLEMENTATION_SUMMARY.md**: Technical details
- **Code**: Well-commented, docstrings included

## 🎯 Next Steps

1. ✅ Test locally (optional)
2. ✅ Submit job
3. ✅ Wait for completion (~minutes to hours)
4. ✅ Check output CSV
5. ✅ Generate visualization
6. ✅ Analyze patterns!

## 📞 Quick Commands

```bash
# Submit job
sbatch pop2vec/llm/slurm_scripts/snellius/build_trie.sh

# Check job
squeue -u $USER
scancel <jobid>  # Cancel if needed

# View logs
tail -f /projects/0/prjs1589/stonybrook/logs/build_trie-*.err
less /projects/0/prjs1589/stonybrook/logs/build_trie-*.out

# Check output
ls -lh /projects/0/prjs1589/stonybrook/fake_data_v0/step5/trie_analysis/
head /projects/0/prjs1589/stonybrook/fake_data_v0/step5/trie_analysis/dryrun_trie.csv

# Visualize
python -m pop2vec.llm.src.new_code.visualize_trie \
    /projects/0/prjs1589/stonybrook/fake_data_v0/step5/trie_analysis/dryrun_trie.csv \
    /projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv \
    -o ~/trie_viz.html

# View (on local machine after scp)
firefox ~/trie_viz.html
```

---

**Ready to go!** All files created and organized. 🎉
