# Enhanced Trie Visualization Guide

## Overview
The enhanced visualization (`trie_viz_enhanced.html`) provides advanced interactive features for exploring life sequence trie data.

## File Location
- **Enhanced HTML**: `/home/ehassan/trie_viz_enhanced.html`
- **Original HTML**: `/home/ehassan/trie_viz.html` (for comparison)

## How to View
Open the HTML file in your web browser:
```bash
# Option 1: From the command line
xdg-open /home/ehassan/trie_viz_enhanced.html

# Option 2: Using Firefox (if available)
firefox /home/ehassan/trie_viz_enhanced.html

# Option 3: Copy to your local machine and open
```

## New Features

### 1. **Multiple Visualization Types**
Switch between different diagram layouts using the dropdown:
- **Tree Layout** (default): Traditional top-down tree structure
- **Radial Layout**: Circular tree radiating from center
- **Sunburst**: Hierarchical donut chart showing proportions
- **Sankey Diagram**: Flow-based visualization showing sequence paths

### 2. **Color Schemes**
Choose how nodes are colored:
- **By Depth**: Color gradient from root to leaves (shows hierarchy depth)
- **By Count**: Heat map based on sequence frequency (hot = common paths)
- **By Category**: Different colors for different token categories (background, events, etc.)

### 3. **Node Size Scaling**
Control how node size represents sequence counts:
- **Linear**: Size proportional to count
- **Square Root**: More balanced sizing (recommended for high variance)
- **Logarithmic**: Compress size differences (good for extreme outliers)

### 4. **Label Display Options**
Toggle label visibility:
- **Always Show**: All node labels visible (can be cluttered)
- **On Hover Only**: Clean view, labels appear on mouseover (recommended)

### 5. **Interactive Controls**
- **Zoom/Pan**: Mouse wheel to zoom, click-drag to pan
- **Reset View**: Button to restore original position/zoom
- **Export SVG**: Save the current view as a vector image
- **Tooltip**: Hover over nodes for detailed statistics:
  - Token name
  - Total count (sequences passing through)
  - End count (sequences ending at this node)
  - Node ID
  - Depth level

### 6. **Statistics Panel**
Real-time statistics displayed in the sidebar:
- Total number of nodes
- Maximum depth of the trie
- Total sequence count
- Number of leaf nodes

## Usage Tips

### For Exploring Common Paths
1. Set color to "By Count"
2. Set size to "Square Root" or "Logarithmic"
3. Use "Tree" or "Sankey" layout
4. Look for bright red nodes (most frequent sequences)

### For Understanding Hierarchy
1. Set color to "By Depth"
2. Use "Radial" or "Tree" layout
3. Enable "Always Show" labels for small sections
4. Zoom in on specific branches

### For Comparing Proportions
1. Use "Sunburst" layout
2. Set color to "By Category" or "By Count"
3. The arc size represents the proportion of sequences

### For Analyzing Flows
1. Use "Sankey" layout
2. Set thickness to "Square Root"
3. Follow the flow paths to see common transitions
4. Wide bands = popular sequence paths

## Technical Details

### Data Loaded
- **Trie CSV**: `/projects/0/prjs1589/stonybrook/visualize/trie_tree/dryrun_trie.csv`
- **Vocabulary**: `/home/ehassan/life-sequencing-dutch/vocab.csv`
- **Max Depth Displayed**: 10 levels
- **Source**: 10,000 test sequences from dry run

### Performance
- The visualization uses D3.js v7 for rendering
- Handles large trie structures efficiently
- SVG-based (vector graphics, infinite zoom quality)
- Client-side only (no server needed)

## Comparing with Original
The original visualization (`trie_viz.html`) is simpler:
- Single tree layout only
- Fixed color scheme
- No size scaling options
- Basic interactivity

The enhanced version adds:
- 4 layout types
- 3 color schemes
- 3 size scaling options
- Label control
- Better UI/UX
- Export functionality

## Next Steps
1. Open the HTML file and explore different combinations
2. Try each visualization type to find what works best for your analysis
3. Use the export feature to save views for presentations
4. Provide feedback on what works and what could be improved

## Customization
To regenerate with different settings:
```bash
python /home/ehassan/life-sequencing-dutch/pop2vec/llm/src/new_code/visualize_trie_enhanced.py \
  /projects/0/prjs1589/stonybrook/visualize/trie_tree/dryrun_trie.csv \
  /home/ehassan/life-sequencing-dutch/vocab.csv \
  --output /home/ehassan/trie_viz_custom.html \
  --title "Custom Title" \
  --max-depth 15 \
  --min-count 5 \
  --max-children 10 \
  --width 1800 \
  --height 1200
```

### Parameters
- `--max-depth N`: Show only first N levels (reduces complexity)
- `--min-count N`: Filter out nodes with count < N (removes rare paths)
- `--max-children N`: Keep only top N children per node by count (reduces clutter) ⭐ **NEW**
- `--width W`: Canvas width in pixels
- `--height H`: Canvas height in pixels
- `--title "Title"`: Custom title for the visualization

### Recommended Settings
- **Quick overview**: `--max-children 5` → ~16K nodes
- **Standard analysis**: `--max-children 10` → ~53K nodes (current default)
- **Detailed view**: `--max-children 20` → ~92K nodes

## Troubleshooting
- **Blank screen**: Check browser console for errors
- **Too cluttered**: Increase `--max-depth` or `--min-count` filters
- **Slow rendering**: Reduce number of nodes with filters
- **Labels overlap**: Use "On Hover Only" mode or zoom in

Enjoy exploring your life sequence data! 🎯
