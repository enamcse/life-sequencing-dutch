# Trie Visualization Comparison - Max Children Filtering

## Summary
The `--max-children` parameter dramatically reduces visual clutter by keeping only the top N most frequent child nodes at each level. This focuses the visualization on the most common sequence paths.

## Generated Visualizations

### 1. Top 5 per node (Most Compact) ✨ **Recommended for Overview**
- **File**: `/home/ehassan/trie_viz_top5.html`
- **Nodes**: 16,076 (99% reduction from 1.6M)
- **Pruned**: 536 branches
- **Best for**: Getting a high-level overview of the most dominant patterns
- **Use cases**:
  - Initial exploration
  - Presentations
  - Identifying main sequence pathways

### 2. Top 10 per node (Balanced) ✨ **Recommended for Analysis**
- **File**: `/home/ehassan/trie_viz_enhanced.html`
- **Nodes**: 53,022 (97% reduction)
- **Pruned**: 492 branches
- **Best for**: Detailed analysis while maintaining clarity
- **Use cases**:
  - Standard analysis
  - Finding common and less-common patterns
  - Balanced detail vs. readability

### 3. Top 20 per node (Detailed)
- **File**: `/home/ehassan/trie_viz_top20.html`
- **Nodes**: 92,261 (95% reduction)
- **Pruned**: 479 branches
- **Best for**: In-depth analysis of diverse patterns
- **Use cases**:
  - Detailed pattern exploration
  - Finding rare but significant paths
  - Research and documentation

### 4. Original (No Filtering) ⚠️ **Not Recommended**
- **File**: `/home/ehassan/trie_viz.html` (old version)
- **Nodes**: 1,676,638 (full dataset)
- **Issues**: Visual flooding, difficult to navigate, slow rendering
- **Only use if**: You need to see ALL possible paths (rare)

## How It Works

The pruning algorithm:
1. **Loads all nodes** from the trie CSV
2. **Builds the hierarchy** with parent-child relationships
3. **Sorts children by count** at each node (descending)
4. **Keeps top N children** per node (N = `--max-children`)
5. **Removes the rest** recursively

### Key Points
- **Sorting by count** ensures most frequent paths are preserved
- **Applied at EVERY level** of the tree
- **Recursive pruning** propagates through the entire structure
- **Statistics preserved** (tooltips still show full counts)

## Usage Examples

### Quick Overview (Top 5)
```bash
python visualize_trie_enhanced.py \
  /projects/0/prjs1589/stonybrook/visualize/trie_tree/dryrun_trie.csv \
  /home/ehassan/life-sequencing-dutch/vocab.csv \
  --output trie_viz_overview.html \
  --max-children 5 \
  --max-depth 8
```

### Standard Analysis (Top 10)
```bash
python visualize_trie_enhanced.py \
  /projects/0/prjs1589/stonybrook/visualize/trie_tree/dryrun_trie.csv \
  /home/ehassan/life-sequencing-dutch/vocab.csv \
  --output trie_viz_analysis.html \
  --max-children 10 \
  --max-depth 10
```

### Detailed Exploration (Top 20)
```bash
python visualize_trie_enhanced.py \
  /projects/0/prjs1589/stonybrook/visualize/trie_tree/dryrun_trie.csv \
  /home/ehassan/life-sequencing-dutch/vocab.csv \
  --output trie_viz_detailed.html \
  --max-children 20 \
  --max-depth 12
```

### With Additional Filters
```bash
python visualize_trie_enhanced.py \
  /projects/0/prjs1589/stonybrook/visualize/trie_tree/dryrun_trie.csv \
  /home/ehassan/life-sequencing-dutch/vocab.csv \
  --output trie_viz_filtered.html \
  --max-children 10 \
  --max-depth 10 \
  --min-count 100  # Only show nodes with 100+ sequences
```

## Visualization Features (All Versions)

All generated visualizations include:
- **4 Layout Types**: Tree, Radial, Sunburst, Sankey
- **3 Color Schemes**: By Depth, By Count, By Category
- **3 Size Scales**: Linear, Square Root, Logarithmic
- **Label Control**: Always show or hover-only
- **Interactive**: Zoom, pan, tooltips
- **Export**: Save as SVG

## Choosing the Right Setting

| Use Case | Max Children | Expected Nodes | Load Time |
|----------|--------------|----------------|-----------|
| Quick overview | 5 | 10K-20K | Fast |
| Standard analysis | 10 | 40K-60K | Medium |
| Detailed research | 20 | 80K-100K | Slower |
| Comprehensive | 30-50 | 150K-200K | Slow |
| Full (not recommended) | None | 1.6M+ | Very slow |

## Tips

### For Best Results
1. **Start with top 5** to get familiar with the structure
2. **Increase gradually** if you need more detail
3. **Use filters together**:
   - `--max-children 10` + `--max-depth 8` = focused view
   - `--max-children 10` + `--min-count 50` = frequent paths only
4. **Match visualization type** to max-children:
   - Sunburst: works well with 5-10
   - Tree/Radial: works well with 10-20
   - Sankey: works well with 5-15

### Understanding the Statistics
- **Total nodes after pruning**: How many nodes are in the visualization
- **Pruned children**: How many branches were removed
- The counts in tooltips are still from the FULL dataset (not affected by pruning)

### Performance Notes
- Top 5: Renders instantly, very smooth interaction
- Top 10: Quick rendering, smooth interaction
- Top 20: Noticeable load time, still interactive
- Top 50+: Longer load, may be sluggish

## Files Generated

```
/home/ehassan/
├── trie_viz_top5.html          # 16K nodes, very clean
├── trie_viz_enhanced.html      # 53K nodes, balanced (default)
├── trie_viz_top20.html         # 92K nodes, detailed
└── trie_viz.html               # 1.6M nodes, original (cluttered)
```

## Next Steps

1. **Open trie_viz_top5.html** for a quick overview
2. **Try different visualization types** (Tree → Radial → Sunburst → Sankey)
3. **Experiment with color schemes** ("By Count" is great for finding hot paths)
4. **Use the enhanced version** (top 10) for your main analysis
5. **Generate custom versions** with different parameters as needed

## Script Location
`/home/ehassan/life-sequencing-dutch/pop2vec/llm/src/new_code/visualize_trie_enhanced.py`

All parameters:
```
--output FILE           Output HTML filename
--title TEXT           Custom title
--max-depth N          Maximum tree depth to show
--min-count N          Minimum sequence count threshold
--max-children N       Maximum children per node (NEW!)
--width W              Canvas width (default: 1400)
--height H             Canvas height (default: 900)
```

Enjoy your decluttered trie visualizations! 🎉
