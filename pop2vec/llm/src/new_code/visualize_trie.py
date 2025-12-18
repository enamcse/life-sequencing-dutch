#!/usr/bin/env python3
"""
Visualize a sequence trie from CSV file.
Generates an interactive D3.js visualization showing common sequence patterns.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_trie_from_csv(csv_path: str) -> pd.DataFrame:
    """Load trie from CSV file"""
    logger.info(f"Loading trie from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} nodes")
    return df


def load_vocabulary(vocab_path: str) -> Dict[int, str]:
    """Load vocabulary and create ID to token mapping"""
    logger.info(f"Loading vocabulary from: {vocab_path}")
    vocab_df = pd.read_csv(vocab_path)
    
    # Handle both upper and lowercase column names
    token_col = 'TOKEN' if 'TOKEN' in vocab_df.columns else 'token'
    id_col = 'ID' if 'ID' in vocab_df.columns else 'id'
    
    id_to_token = dict(zip(vocab_df[id_col], vocab_df[token_col]))
    logger.info(f"Loaded {len(id_to_token):,} tokens")
    
    return id_to_token


def trie_to_d3_json(
    trie_df: pd.DataFrame,
    id_to_token: Dict[int, str],
    max_depth: Optional[int] = None,
    min_count: Optional[int] = None
) -> Dict:
    """
    Convert trie DataFrame to D3.js hierarchical format.
    
    Args:
        trie_df: DataFrame with trie nodes
        id_to_token: Mapping from token ID to token string
        max_depth: Maximum depth to include (None = all)
        min_count: Minimum count to include (None = all)
    
    Returns:
        Dictionary in D3 hierarchy format
    """
    logger.info("Converting trie to D3 format...")
    
    # Filter if needed
    if min_count is not None:
        trie_df = trie_df[trie_df['count'] >= min_count]
        logger.info(f"Filtered to {len(trie_df):,} nodes with count >= {min_count}")
    
    # Parse child_list JSON strings
    trie_df['child_dict'] = trie_df['child_list'].apply(lambda x: json.loads(x) if pd.notna(x) else {})
    
    # Create node lookup
    nodes = {}
    for _, row in trie_df.iterrows():
        node_id = int(row['node_id'])
        token_id = int(row['token'])
        token_str = id_to_token.get(token_id, f"TOKEN_{token_id}")
        
        nodes[node_id] = {
            'node_id': node_id,
            'name': token_str,
            'token_id': token_id,
            'count': int(row['count']),
            'end_count': int(row['end_count']),
            'parent_id': int(row['parent']),
            'children': []
        }
    
    # Build hierarchy
    root = None
    for node_id, node in nodes.items():
        parent_id = node['parent_id']
        if parent_id == -1:
            root = node
        elif parent_id in nodes:
            nodes[parent_id]['children'].append(node)
    
    if root is None:
        raise ValueError("No root node found in trie")
    
    # Limit depth if specified
    if max_depth is not None:
        def limit_depth(node: Dict, depth: int = 0):
            if depth >= max_depth:
                node['children'] = []
            else:
                for child in node['children']:
                    limit_depth(child, depth + 1)
        
        limit_depth(root)
    
    logger.info(f"Created D3 hierarchy with root: {root['name']}")
    
    return root


def generate_html_visualization(
    d3_data: Dict,
    output_path: str,
    title: str = "Sequence Trie Visualization",
    width: int = 1200,
    height: int = 800
) -> None:
    """
    Generate an HTML file with interactive D3.js visualization.
    
    Args:
        d3_data: Hierarchy data in D3 format
        output_path: Path to output HTML file
        title: Title for the visualization
        width: Width of the visualization
        height: Height of the visualization
    """
    logger.info(f"Generating HTML visualization: {output_path}")
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        h1 {{
            text-align: center;
            color: #333;
        }}
        
        #controls {{
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        #controls label {{
            margin: 0 10px;
            font-weight: 600;
        }}
        
        #controls input, #controls select {{
            margin: 0 5px;
            padding: 5px;
        }}
        
        #visualization {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 20px auto;
            width: {width}px;
            height: {height}px;
        }}
        
        .node circle {{
            cursor: pointer;
            stroke: #3182bd;
            stroke-width: 2px;
        }}
        
        .node text {{
            font-size: 12px;
            font-family: 'Courier New', monospace;
        }}
        
        .link {{
            fill: none;
            stroke: #ccc;
            stroke-width: 2px;
        }}
        
        .tooltip {{
            position: absolute;
            text-align: left;
            padding: 10px;
            font-size: 12px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 4px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
        }}
        
        #stats {{
            text-align: center;
            margin: 20px;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        #stats .stat {{
            display: inline-block;
            margin: 0 20px;
        }}
        
        #stats .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3182bd;
        }}
        
        #stats .stat-label {{
            font-size: 14px;
            color: #666;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    
    <div id="controls">
        <label for="minCount">Min Count:</label>
        <input type="number" id="minCount" value="10" min="1" max="1000">
        
        <label for="maxDepth">Max Depth:</label>
        <input type="number" id="maxDepth" value="10" min="1" max="50">
        
        <label for="layout">Layout:</label>
        <select id="layout">
            <option value="tree">Tree</option>
            <option value="cluster">Cluster</option>
            <option value="radial">Radial</option>
        </select>
        
        <button onclick="updateVisualization()">Update</button>
        <button onclick="resetZoom()">Reset Zoom</button>
    </div>
    
    <div id="stats">
        <div class="stat">
            <div class="stat-value" id="totalNodes">-</div>
            <div class="stat-label">Total Nodes</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="maxDepth">-</div>
            <div class="stat-label">Max Depth</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="totalSequences">-</div>
            <div class="stat-label">Total Sequences</div>
        </div>
    </div>
    
    <div id="visualization"></div>
    
    <script>
        // Trie data
        const trieData = {json.dumps(d3_data, indent=2)};
        
        const width = {width};
        const height = {height};
        let svg, g, zoom, root;
        
        // Initialize visualization
        function initVisualization() {{
            // Create SVG
            svg = d3.select("#visualization")
                .append("svg")
                .attr("width", width)
                .attr("height", height);
            
            // Create zoom behavior
            zoom = d3.zoom()
                .scaleExtent([0.1, 4])
                .on("zoom", (event) => {{
                    g.attr("transform", event.transform);
                }});
            
            svg.call(zoom);
            
            // Create group for content
            g = svg.append("g");
            
            // Create tooltip
            d3.select("body").append("div")
                .attr("class", "tooltip")
                .attr("id", "tooltip");
            
            updateVisualization();
        }}
        
        function updateVisualization() {{
            const minCount = +document.getElementById("minCount").value;
            const maxDepth = +document.getElementById("maxDepth").value;
            const layout = document.getElementById("layout").value;
            
            // Filter data
            let filteredData = JSON.parse(JSON.stringify(trieData));
            filterByCount(filteredData, minCount);
            limitDepth(filteredData, maxDepth);
            
            // Calculate statistics
            const stats = calculateStats(filteredData);
            document.getElementById("totalNodes").textContent = stats.totalNodes.toLocaleString();
            document.getElementById("maxDepth").textContent = stats.maxDepth;
            document.getElementById("totalSequences").textContent = stats.totalSequences.toLocaleString();
            
            // Clear existing visualization
            g.selectAll("*").remove();
            
            // Create hierarchy
            root = d3.hierarchy(filteredData);
            
            // Choose layout
            let treeLayout;
            if (layout === "radial") {{
                treeLayout = d3.tree()
                    .size([2 * Math.PI, Math.min(width, height) / 3])
                    .separation((a, b) => (a.parent == b.parent ? 1 : 2) / a.depth);
                
                treeLayout(root);
                
                // Convert to radial coordinates
                root.descendants().forEach(d => {{
                    d.x0 = d.x;
                    d.y0 = d.y;
                    d.x = d.y * Math.cos(d.x - Math.PI / 2);
                    d.y = d.y * Math.sin(d.x0 - Math.PI / 2);
                }});
            }} else if (layout === "cluster") {{
                treeLayout = d3.cluster()
                    .size([height - 100, width - 200]);
                treeLayout(root);
            }} else {{
                treeLayout = d3.tree()
                    .size([height - 100, width - 200]);
                treeLayout(root);
            }}
            
            // Draw links
            g.selectAll(".link")
                .data(root.links())
                .enter()
                .append("path")
                .attr("class", "link")
                .attr("d", layout === "radial" ? 
                    d3.linkRadial()
                        .angle(d => d.x0)
                        .radius(d => d.y0) :
                    d3.linkHorizontal()
                        .x(d => d.y)
                        .y(d => d.x)
                );
            
            // Draw nodes
            const node = g.selectAll(".node")
                .data(root.descendants())
                .enter()
                .append("g")
                .attr("class", "node")
                .attr("transform", d => `translate(${{d.y}},${{d.x}})`);
            
            node.append("circle")
                .attr("r", d => Math.max(3, Math.min(15, Math.sqrt(d.data.count) / 2)))
                .style("fill", d => {{
                    const intensity = Math.log(d.data.count + 1) / Math.log(stats.totalSequences);
                    return d3.interpolateBlues(0.3 + intensity * 0.7);
                }})
                .on("mouseover", function(event, d) {{
                    d3.select("#tooltip")
                        .style("opacity", 1)
                        .html(`
                            <strong>${{d.data.name}}</strong><br>
                            Count: ${{d.data.count.toLocaleString()}}<br>
                            End Count: ${{d.data.end_count.toLocaleString()}}<br>
                            Depth: ${{d.depth}}
                        `)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                }})
                .on("mouseout", function() {{
                    d3.select("#tooltip").style("opacity", 0);
                }});
            
            node.append("text")
                .attr("dy", ".35em")
                .attr("x", d => d.children ? -20 : 20)
                .style("text-anchor", d => d.children ? "end" : "start")
                .text(d => d.depth === 0 ? "ROOT" : d.data.name)
                .style("font-size", d => d.depth === 0 ? "14px" : "11px")
                .style("font-weight", d => d.depth === 0 ? "bold" : "normal");
            
            // Center the tree
            const bounds = g.node().getBBox();
            const dx = width / 2 - bounds.x - bounds.width / 2;
            const dy = height / 2 - bounds.y - bounds.height / 2;
            
            svg.call(zoom.transform, d3.zoomIdentity.translate(dx, dy));
        }}
        
        function filterByCount(node, minCount) {{
            if (!node.children) return;
            node.children = node.children.filter(child => child.count >= minCount);
            node.children.forEach(child => filterByCount(child, minCount));
        }}
        
        function limitDepth(node, maxDepth, depth = 0) {{
            if (depth >= maxDepth) {{
                node.children = [];
            }} else if (node.children) {{
                node.children.forEach(child => limitDepth(child, maxDepth, depth + 1));
            }}
        }}
        
        function calculateStats(node) {{
            let totalNodes = 0;
            let maxDepth = 0;
            let totalSequences = node.count;
            
            function traverse(n, depth) {{
                totalNodes++;
                maxDepth = Math.max(maxDepth, depth);
                if (n.children) {{
                    n.children.forEach(child => traverse(child, depth + 1));
                }}
            }}
            
            traverse(node, 0);
            
            return {{ totalNodes, maxDepth, totalSequences }};
        }}
        
        function resetZoom() {{
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity);
        }}
        
        // Initialize on load
        initVisualization();
    </script>
</body>
</html>"""
    
    with open(output_path, 'w') as f:
        f.write(html_template)
    
    logger.info(f"Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize sequence trie")
    parser.add_argument("trie_csv", help="Path to trie CSV file")
    parser.add_argument("vocab_csv", help="Path to vocabulary CSV file")
    parser.add_argument("--output", "-o", help="Output HTML file", default="trie_visualization.html")
    parser.add_argument("--title", "-t", help="Visualization title", default="Sequence Trie Visualization")
    parser.add_argument("--max-depth", type=int, help="Maximum depth to visualize")
    parser.add_argument("--min-count", type=int, help="Minimum count to include")
    parser.add_argument("--width", type=int, default=1400, help="Visualization width")
    parser.add_argument("--height", type=int, default=900, help="Visualization height")
    
    args = parser.parse_args()
    
    # Load data
    trie_df = load_trie_from_csv(args.trie_csv)
    id_to_token = load_vocabulary(args.vocab_csv)
    
    # Convert to D3 format
    d3_data = trie_to_d3_json(
        trie_df,
        id_to_token,
        max_depth=args.max_depth,
        min_count=args.min_count
    )
    
    # Generate HTML
    generate_html_visualization(
        d3_data,
        args.output,
        title=args.title,
        width=args.width,
        height=args.height
    )
    
    logger.info("Done! Open the HTML file in a web browser to view the visualization.")


if __name__ == "__main__":
    main()
