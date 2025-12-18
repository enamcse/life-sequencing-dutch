#!/usr/bin/env python3
"""
Enhanced visualization for sequence trie from CSV file.
Features:
- Better use of color (depth-based color scale)
- Node size based on count
- Toggle for labels (show on hover only or always)
- Multiple visualization types: Tree, Radial, Sunburst, Sankey
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    if 'TOKEN' in vocab_df.columns and 'ID' in vocab_df.columns:
        token_col, id_col = 'TOKEN', 'ID'
    elif 'token' in vocab_df.columns and 'token_id' in vocab_df.columns:
        token_col, id_col = 'token', 'token_id'
    elif 'token' in vocab_df.columns and 'id' in vocab_df.columns:
        token_col, id_col = 'token', 'id'
    else:
        raise ValueError(f"Could not find token/id columns in vocabulary. Available columns: {vocab_df.columns.tolist()}")
    
    id_to_token = dict(zip(vocab_df[id_col], vocab_df[token_col]))
    logger.info(f"Loaded {len(id_to_token):,} tokens")
    
    return id_to_token


def trie_to_d3_json(
    trie_df: pd.DataFrame,
    id_to_token: Dict[int, str],
    max_depth: Optional[int] = None,
    min_count: Optional[int] = None,
    max_children: Optional[int] = None
) -> Dict:
    """
    Convert trie DataFrame to D3.js hierarchical format.
    
    Args:
        trie_df: DataFrame containing trie nodes
        id_to_token: Mapping from token IDs to token strings
        max_depth: Maximum depth to include in visualization
        min_count: Minimum count threshold for nodes
        max_children: Maximum number of children to keep per node (keeps top N by count)
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
    
    # Prune: keep only top N children per node (sorted by count)
    # BUT: Make sure we do this AFTER depth limiting to preserve deep branches
    if max_children is not None:
        # First, limit depth to get the tree we want to visualize
        if max_depth is not None:
            def limit_depth_first(node: Dict, depth: int = 0):
                if depth >= max_depth:
                    node['children'] = []
                else:
                    for child in node['children']:
                        limit_depth_first(child, depth + 1)
            
            limit_depth_first(root)
            logger.info(f"Limited depth to {max_depth} before pruning")
        
        # Now prune children
        pruned_count = 0
        total_nodes = 0
        
        def prune_children(node: Dict):
            nonlocal pruned_count, total_nodes
            total_nodes += 1
            
            if len(node['children']) > max_children:
                # Sort children by count (descending) and keep top N
                node['children'].sort(key=lambda x: x['count'], reverse=True)
                pruned = len(node['children']) - max_children
                pruned_count += pruned
                node['children'] = node['children'][:max_children]
            
            # Recursively prune children
            for child in node['children']:
                prune_children(child)
        
        prune_children(root)
        logger.info(f"Pruned {pruned_count:,} children (kept top {max_children} per node)")
        logger.info(f"Total nodes after pruning: {total_nodes:,}")
    elif max_depth is not None:
        # Only apply depth limiting if no children pruning was done
        def limit_depth_only(node: Dict, depth: int = 0):
            if depth >= max_depth:
                node['children'] = []
            else:
                for child in node['children']:
                    limit_depth_only(child, depth + 1)
        
        limit_depth_only(root)
    
    logger.info(f"Created D3 hierarchy with root: {root['name']}")
    
    return root


def generate_html_visualization(
    d3_data: Dict,
    output_path: str,
    title: str = "Sequence Trie Visualization - Enhanced",
    width: int = 1400,
    height: int = 900
) -> None:
    """
    Generate an enhanced HTML file with interactive D3.js visualization.
    """
    logger.info(f"Generating enhanced HTML visualization: {output_path}")
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3-sankey@0.12.3/dist/d3-sankey.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: {width + 100}px;
            margin: 0 auto;
        }}
        
        h1 {{
            text-align: center;
            color: white;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .subtitle {{
            text-align: center;
            color: rgba(255,255,255,0.9);
            margin-bottom: 20px;
            font-size: 14px;
        }}
        
        #controls {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .control-group {{
            display: flex;
            flex-direction: column;
        }}
        
        .control-group label {{
            font-weight: 600;
            margin-bottom: 5px;
            color: #333;
            font-size: 13px;
        }}
        
        .control-group input,
        .control-group select {{
            padding: 8px 12px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 14px;
            transition: all 0.3s;
        }}
        
        .control-group input:focus,
        .control-group select:focus {{
            outline: none;
            border-color: #667eea;
        }}
        
        .checkbox-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .checkbox-group input[type="checkbox"] {{
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}
        
        .button-group {{
            display: flex;
            gap: 10px;
            grid-column: 1 / -1;
        }}
        
        button {{
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            flex: 1;
        }}
        
        button.primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        button.primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        
        button.secondary {{
            background: #f0f0f0;
            color: #333;
        }}
        
        button.secondary:hover {{
            background: #e0e0e0;
        }}
        
        #visualization {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            overflow: hidden;
            position: relative;
        }}
        
        #visualization svg {{
            display: block;
        }}
        
        .node circle {{
            cursor: pointer;
            stroke: white;
            stroke-width: 2px;
            transition: all 0.3s;
        }}
        
        .node:hover circle {{
            stroke-width: 4px;
            filter: brightness(1.1);
        }}
        
        .node text {{
            font-family: 'Courier New', monospace;
            pointer-events: none;
            text-shadow: 0 1px 2px rgba(255,255,255,0.8);
        }}
        
        .link {{
            fill: none;
            stroke-opacity: 0.4;
            transition: stroke-opacity 0.3s;
        }}
        
        .link:hover {{
            stroke-opacity: 0.7;
        }}
        
        .tooltip {{
            position: absolute;
            text-align: left;
            padding: 12px;
            font-size: 13px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            border-radius: 8px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            max-width: 300px;
            z-index: 1000;
        }}
        
        .tooltip strong {{
            color: #667eea;
            font-size: 14px;
        }}
        
        #stats {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin-top: 20px;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
        }}
        
        .stat {{
            text-align: center;
            min-width: 150px;
        }}
        
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .stat-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        
        .legend {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            font-size: 12px;
        }}
        
        .legend-title {{
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            margin-right: 8px;
            border: 1px solid #ddd;
        }}
        
        /* Sankey specific styles */
        .sankey-link {{
            fill: none;
            stroke-opacity: 0.5;
        }}
        
        .sankey-link:hover {{
            stroke-opacity: 0.7;
        }}
        
        .sankey-node rect {{
            stroke: white;
            stroke-width: 2px;
        }}
        
        .sankey-node text {{
            pointer-events: none;
            text-shadow: 0 1px 0 #fff;
        }}
        
        /* Sunburst specific styles */
        .sunburst-arc {{
            cursor: pointer;
            stroke: white;
            stroke-width: 2px;
        }}
        
        .sunburst-arc:hover {{
            opacity: 0.8;
        }}
        
        .sunburst-text {{
            pointer-events: none;
            text-anchor: middle;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="subtitle">Interactive exploration of life sequence patterns</div>
        
        <div id="controls">
            <div class="control-group">
                <label for="vizType">Visualization Type</label>
                <select id="vizType" onchange="updateVisualization()">
                    <option value="tree">Tree Layout</option>
                    <option value="radial">Radial Tree</option>
                    <option value="sunburst">Sunburst</option>
                    <option value="sankey">Sankey Diagram</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="minCount">Minimum Count</label>
                <input type="number" id="minCount" value="10" min="1" max="1000" onchange="updateVisualization()">
            </div>
            
            <div class="control-group">
                <label for="maxDepth">Maximum Depth</label>
                <input type="number" id="maxDepth" value="10" min="1" max="50" onchange="updateVisualization()">
            </div>
            
            <div class="control-group">
                <label for="nodeSize">Node Size Scale</label>
                <select id="nodeSize" onchange="updateVisualization()">
                    <option value="linear">Linear</option>
                    <option value="sqrt" selected>Square Root</option>
                    <option value="log">Logarithmic</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="colorScheme">Color Scheme</label>
                <select id="colorScheme" onchange="updateVisualization()">
                    <option value="depth">By Depth</option>
                    <option value="count">By Count</option>
                    <option value="category">By Category</option>
                </select>
            </div>
            
            <div class="control-group checkbox-group">
                <input type="checkbox" id="showLabels" checked onchange="updateVisualization()">
                <label for="showLabels">Always Show Labels</label>
            </div>
            
            <div class="button-group">
                <button class="primary" onclick="updateVisualization()">🔄 Update</button>
                <button class="secondary" onclick="resetZoom()">🔍 Reset View</button>
                <button class="secondary" onclick="downloadSVG()">💾 Download SVG</button>
            </div>
        </div>
        
        <div id="visualization">
            <div class="legend" id="legend"></div>
        </div>
        
        <div id="stats">
            <div class="stat">
                <div class="stat-value" id="totalNodes">-</div>
                <div class="stat-label">Total Nodes</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="maxDepthStat">-</div>
                <div class="stat-label">Maximum Depth</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="totalSequences">-</div>
                <div class="stat-label">Total Sequences</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="avgBranching">-</div>
                <div class="stat-label">Avg Branching</div>
            </div>
        </div>
    </div>
    
    <script>
        // Trie data
        const trieData = {json.dumps(d3_data, indent=2)};
        
        const width = {width};
        const height = {height};
        let svg, g, zoom, root, currentVizType;
        
        // Color schemes
        const colorSchemes = {{
            depth: d3.scaleSequential(d3.interpolateViridis),
            count: d3.scaleSequential(d3.interpolatePlasma),
            category: d3.scaleOrdinal(d3.schemeCategory10)
        }};
        
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
                    if (g) g.attr("transform", event.transform);
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
        
        function getNodeColor(d, scheme, maxDepth, maxCount) {{
            const colorScale = colorSchemes[scheme];
            
            if (scheme === 'depth') {{
                return colorScale(d.depth / maxDepth);
            }} else if (scheme === 'count') {{
                return colorScale(Math.log(d.data.count + 1) / Math.log(maxCount + 1));
            }} else {{
                // Category based on token prefix or type
                const category = d.data.name.split('_')[0];
                return colorScale(category);
            }}
        }}
        
        function getNodeSize(d, sizeScale, maxCount) {{
            const count = d.data.count;
            if (sizeScale === 'linear') {{
                return Math.max(3, Math.min(25, count / maxCount * 25));
            }} else if (sizeScale === 'log') {{
                return Math.max(3, Math.min(25, Math.log(count + 1) * 3));
            }} else {{ // sqrt
                return Math.max(3, Math.min(25, Math.sqrt(count) * 0.5));
            }}
        }}
        
        function updateVisualization() {{
            const minCount = +document.getElementById("minCount").value;
            const maxDepth = +document.getElementById("maxDepth").value;
            const vizType = document.getElementById("vizType").value;
            const nodeSize = document.getElementById("nodeSize").value;
            const colorScheme = document.getElementById("colorScheme").value;
            const showLabels = document.getElementById("showLabels").checked;
            
            currentVizType = vizType;
            
            // Filter data
            let filteredData = JSON.parse(JSON.stringify(trieData));
            filterByCount(filteredData, minCount);
            limitDepth(filteredData, maxDepth);
            
            // Calculate statistics
            const stats = calculateStats(filteredData);
            document.getElementById("totalNodes").textContent = stats.totalNodes.toLocaleString();
            document.getElementById("maxDepthStat").textContent = stats.maxDepth;
            document.getElementById("totalSequences").textContent = stats.totalSequences.toLocaleString();
            document.getElementById("avgBranching").textContent = stats.avgBranching.toFixed(2);
            
            // Clear existing visualization
            g.selectAll("*").remove();
            
            // Update legend
            updateLegend(colorScheme, stats.maxDepth, stats.totalSequences);
            
            // Render based on type
            if (vizType === 'sankey') {{
                renderSankey(filteredData, nodeSize, colorScheme, stats);
            }} else if (vizType === 'sunburst') {{
                renderSunburst(filteredData, nodeSize, colorScheme, showLabels, stats);
            }} else {{
                renderTree(filteredData, vizType === 'radial', nodeSize, colorScheme, showLabels, stats);
            }}
        }}
        
        function renderTree(data, isRadial, nodeSize, colorScheme, showLabels, stats) {{
            root = d3.hierarchy(data);
            
            let treeLayout;
            if (isRadial) {{
                treeLayout = d3.tree()
                    .size([2 * Math.PI, Math.min(width, height) / 2.5])
                    .separation((a, b) => (a.parent == b.parent ? 1 : 2) / a.depth);
                
                treeLayout(root);
                
                // Convert to radial coordinates
                root.descendants().forEach(d => {{
                    d.x0 = d.x;
                    d.y0 = d.y;
                    d.x = d.y * Math.cos(d.x0 - Math.PI / 2);
                    d.y = d.y * Math.sin(d.x0 - Math.PI / 2);
                }});
            }} else {{
                treeLayout = d3.tree()
                    .size([height - 100, width - 250]);
                treeLayout(root);
            }}
            
            // Draw links with color gradient
            const linkGen = isRadial ? 
                d3.linkRadial().angle(d => d.x0).radius(d => d.y0) :
                d3.linkHorizontal().x(d => d.y).y(d => d.x);
            
            g.selectAll(".link")
                .data(root.links())
                .enter()
                .append("path")
                .attr("class", "link")
                .attr("d", linkGen)
                .style("stroke", d => getNodeColor(d.target, colorScheme, stats.maxDepth, stats.totalSequences))
                .style("stroke-width", d => Math.max(1, Math.sqrt(d.target.data.count) * 0.1));
            
            // Draw nodes
            const node = g.selectAll(".node")
                .data(root.descendants())
                .enter()
                .append("g")
                .attr("class", "node")
                .attr("transform", d => isRadial ? 
                    `translate(${{d.x}},${{d.y}})` : 
                    `translate(${{d.y}},${{d.x}})`);
            
            node.append("circle")
                .attr("r", d => getNodeSize(d, nodeSize, stats.totalSequences))
                .style("fill", d => getNodeColor(d, colorScheme, stats.maxDepth, stats.totalSequences))
                .on("mouseover", showTooltip)
                .on("mouseout", hideTooltip);
            
            // Add labels
            const textElement = node.append("text")
                .attr("dy", ".35em")
                .attr("x", d => {{
                    if (isRadial) {{
                        return d.x0 < Math.PI ? 10 : -10;
                    }} else {{
                        return d.children ? -10 : 10;
                    }}
                }})
                .style("text-anchor", d => {{
                    if (isRadial) {{
                        return d.x0 < Math.PI ? "start" : "end";
                    }} else {{
                        return d.children ? "end" : "start";
                    }}
                }})
                .text(d => d.depth === 0 ? "ROOT" : d.data.name)
                .style("font-size", d => d.depth === 0 ? "14px" : "11px")
                .style("font-weight", d => d.depth === 0 ? "bold" : "normal")
                .style("fill", "#333");
            
            if (!showLabels) {{
                textElement.style("opacity", 0);
                node.on("mouseenter", function() {{
                    d3.select(this).select("text").style("opacity", 1);
                }})
                .on("mouseleave", function() {{
                    d3.select(this).select("text").style("opacity", 0);
                }});
            }}
            
            // Center the tree
            centerView();
        }}
        
        function renderSunburst(data, nodeSize, colorScheme, showLabels, stats) {{
            root = d3.hierarchy(data)
                .sum(d => d.count)
                .sort((a, b) => b.value - a.value);
            
            const radius = Math.min(width, height) / 2 - 10;
            
            const partition = d3.partition()
                .size([2 * Math.PI, radius]);
            
            partition(root);
            
            const arc = d3.arc()
                .startAngle(d => d.x0)
                .endAngle(d => d.x1)
                .innerRadius(d => d.y0)
                .outerRadius(d => d.y1);
            
            g.attr("transform", `translate(${{width / 2}},${{height / 2}})`);
            
            const path = g.selectAll("path")
                .data(root.descendants())
                .enter()
                .append("path")
                .attr("class", "sunburst-arc")
                .attr("d", arc)
                .style("fill", d => getNodeColor(d, colorScheme, stats.maxDepth, stats.totalSequences))
                .style("opacity", 0.8)
                .on("mouseover", function(event, d) {{
                    d3.select(this).style("opacity", 1);
                    showTooltip(event, d);
                }})
                .on("mouseout", function(event, d) {{
                    d3.select(this).style("opacity", 0.8);
                    hideTooltip(event, d);
                }});
            
            if (showLabels) {{
                g.selectAll("text")
                    .data(root.descendants().filter(d => d.depth > 0 && (d.x1 - d.x0) > 0.05))
                    .enter()
                    .append("text")
                    .attr("class", "sunburst-text")
                    .attr("transform", d => {{
                        const angle = (d.x0 + d.x1) / 2;
                        const r = (d.y0 + d.y1) / 2;
                        return `rotate(${{angle * 180 / Math.PI - 90}}) translate(${{r}},0) rotate(${{angle > Math.PI ? 180 : 0}})`;
                    }})
                    .attr("dy", "0.35em")
                    .text(d => d.data.name.length > 8 ? d.data.name.substring(0, 8) + "..." : d.data.name)
                    .style("fill", "white")
                    .style("font-size", "9px");
            }}
        }}
        
        function renderSankey(data, nodeSize, colorScheme, stats) {{
            // Convert tree to Sankey links
            const nodes = [];
            const links = [];
            const nodeMap = new Map();
            
            function traverse(node, depth = 0) {{
                const id = `${{depth}}-${{node.name}}`;
                if (!nodeMap.has(id)) {{
                    nodeMap.set(id, nodes.length);
                    nodes.push({{
                        name: node.name,
                        count: node.count,
                        depth: depth
                    }});
                }}
                
                if (node.children) {{
                    node.children.forEach(child => {{
                        const childId = `${{depth + 1}}-${{child.name}}`;
                        if (!nodeMap.has(childId)) {{
                            nodeMap.set(childId, nodes.length);
                            nodes.push({{
                                name: child.name,
                                count: child.count,
                                depth: depth + 1
                            }});
                        }}
                        
                        links.push({{
                            source: nodeMap.get(id),
                            target: nodeMap.get(childId),
                            value: child.count
                        }});
                        
                        traverse(child, depth + 1);
                    }});
                }}
            }}
            
            traverse(data);
            
            // Create Sankey diagram
            const sankey = d3.sankey()
                .nodeWidth(15)
                .nodePadding(10)
                .extent([[10, 10], [width - 10, height - 50]]);
            
            const graph = sankey({{
                nodes: nodes.map(d => Object.assign({{}}, d)),
                links: links.map(d => Object.assign({{}}, d))
            }});
            
            // Draw links
            g.selectAll(".sankey-link")
                .data(graph.links)
                .enter()
                .append("path")
                .attr("class", "sankey-link")
                .attr("d", d3.sankeyLinkHorizontal())
                .style("stroke", d => getNodeColor({{depth: d.target.depth, data: d.target}}, 
                    colorScheme, stats.maxDepth, stats.totalSequences))
                .style("stroke-width", d => Math.max(1, d.width))
                .on("mouseover", function(event, d) {{
                    showTooltip(event, {{data: d.target}});
                }})
                .on("mouseout", hideTooltip);
            
            // Draw nodes
            const node = g.selectAll(".sankey-node")
                .data(graph.nodes)
                .enter()
                .append("g")
                .attr("class", "sankey-node");
            
            node.append("rect")
                .attr("x", d => d.x0)
                .attr("y", d => d.y0)
                .attr("height", d => d.y1 - d.y0)
                .attr("width", d => d.x1 - d.x0)
                .style("fill", d => getNodeColor({{depth: d.depth, data: d}}, 
                    colorScheme, stats.maxDepth, stats.totalSequences))
                .on("mouseover", function(event, d) {{
                    showTooltip(event, {{data: d}});
                }})
                .on("mouseout", hideTooltip);
            
            node.append("text")
                .attr("x", d => d.x0 < width / 2 ? d.x1 + 6 : d.x0 - 6)
                .attr("y", d => (d.y1 + d.y0) / 2)
                .attr("dy", "0.35em")
                .attr("text-anchor", d => d.x0 < width / 2 ? "start" : "end")
                .text(d => d.name)
                .style("font-size", "11px")
                .style("fill", "#333");
        }}
        
        function showTooltip(event, d) {{
            d3.select("#tooltip")
                .style("opacity", 1)
                .html(`
                    <strong>${{d.data.name}}</strong><br>
                    Count: ${{d.data.count.toLocaleString()}}<br>
                    End Count: ${{d.data.end_count.toLocaleString()}}<br>
                    Depth: ${{d.depth || 0}}<br>
                    Percentage: ${{((d.data.count / trieData.count) * 100).toFixed(2)}}%
                `)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        }}
        
        function hideTooltip() {{
            d3.select("#tooltip").style("opacity", 0);
        }}
        
        function updateLegend(colorScheme, maxDepth, maxCount) {{
            const legend = d3.select("#legend");
            legend.html("");
            
            legend.append("div")
                .attr("class", "legend-title")
                .text("Color Scale: " + colorScheme.charAt(0).toUpperCase() + colorScheme.slice(1));
            
            if (colorScheme === 'depth') {{
                for (let i = 0; i <= 5; i++) {{
                    const item = legend.append("div").attr("class", "legend-item");
                    item.append("div")
                        .attr("class", "legend-color")
                        .style("background-color", colorSchemes.depth(i / 5));
                    item.append("span").text(`Depth ${{Math.round(i * maxDepth / 5)}}`);
                }}
            }} else if (colorScheme === 'count') {{
                for (let i = 0; i <= 5; i++) {{
                    const item = legend.append("div").attr("class", "legend-item");
                    item.append("div")
                        .attr("class", "legend-color")
                        .style("background-color", colorSchemes.count(i / 5));
                    const count = Math.round(Math.exp((i / 5) * Math.log(maxCount + 1)));
                    item.append("span").text(`~${{count.toLocaleString()}}`);
                }}
            }}
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
            let totalChildren = 0;
            let nodesWithChildren = 0;
            const totalSequences = node.count;
            
            function traverse(n, depth) {{
                totalNodes++;
                maxDepth = Math.max(maxDepth, depth);
                if (n.children && n.children.length > 0) {{
                    totalChildren += n.children.length;
                    nodesWithChildren++;
                    n.children.forEach(child => traverse(child, depth + 1));
                }}
            }}
            
            traverse(node, 0);
            
            const avgBranching = nodesWithChildren > 0 ? totalChildren / nodesWithChildren : 0;
            
            return {{ totalNodes, maxDepth, totalSequences, avgBranching }};
        }}
        
        function centerView() {{
            const bounds = g.node().getBBox();
            const dx = width / 2 - bounds.x - bounds.width / 2;
            const dy = height / 2 - bounds.y - bounds.height / 2;
            
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity.translate(dx, dy));
        }}
        
        function resetZoom() {{
            if (currentVizType === 'sunburst') {{
                svg.transition()
                    .duration(750)
                    .call(zoom.transform, d3.zoomIdentity);
            }} else {{
                centerView();
            }}
        }}
        
        function downloadSVG() {{
            const svgElement = document.querySelector("#visualization svg");
            const serializer = new XMLSerializer();
            const svgString = serializer.serializeToString(svgElement);
            const blob = new Blob([svgString], {{type: "image/svg+xml"}});
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = "trie_visualization.svg";
            link.click();
            URL.revokeObjectURL(url);
        }}
        
        // Initialize on load
        initVisualization();
    </script>
</body>
</html>"""
    
    with open(output_path, 'w') as f:
        f.write(html_template)
    
    logger.info(f"Saved enhanced visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced visualization for sequence trie")
    parser.add_argument("trie_csv", help="Path to trie CSV file")
    parser.add_argument("vocab_csv", help="Path to vocabulary CSV file")
    parser.add_argument("--output", "-o", help="Output HTML file", default="trie_viz_enhanced.html")
    parser.add_argument("--title", "-t", help="Visualization title", 
                        default="Sequence Trie Visualization - Enhanced")
    parser.add_argument("--max-depth", type=int, help="Maximum depth to visualize")
    parser.add_argument("--min-count", type=int, help="Minimum count to include")
    parser.add_argument("--max-children", type=int, help="Maximum number of children per node (keeps top N by count)")
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
        min_count=args.min_count,
        max_children=args.max_children
    )
    
    # Generate HTML
    generate_html_visualization(
        d3_data,
        args.output,
        title=args.title,
        width=args.width,
        height=args.height
    )
    
    logger.info("Done! Open the HTML file in a web browser to view the enhanced visualization.")
    logger.info(f"Features: Tree/Radial/Sunburst/Sankey views, color by depth/count, size scaling, label toggle")
    if args.max_children:
        logger.info(f"Limited to top {args.max_children} children per node by count")


if __name__ == "__main__":
    main()
