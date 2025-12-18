#!/usr/bin/env python3
import pandas as pd
import json

# Load trie
df = pd.read_csv('/projects/0/prjs1589/stonybrook/visualize/trie_tree/dryrun_trie.csv')
print(f'Total nodes: {len(df):,}')

# Parse child lists
df['child_dict'] = df['child_list'].apply(lambda x: json.loads(x) if pd.notna(x) else {})

# Build hierarchy
nodes = {}
for _, row in df.iterrows():
    node_id = int(row['node_id'])
    nodes[node_id] = {
        'node_id': node_id,
        'count': int(row['count']),
        'parent_id': int(row['parent']),
        'children': []
    }

# Connect children
for node_id, node in nodes.items():
    parent_id = node['parent_id']
    if parent_id != -1 and parent_id in nodes:
        nodes[parent_id]['children'].append(node)

# Find root
root = nodes[0]

# Calculate depth before pruning
def calc_depth(node, depth=0):
    if not node['children']:
        return depth
    return max(calc_depth(child, depth+1) for child in node['children'])

print(f'Max depth BEFORE pruning: {calc_depth(root)}')

# Prune to top 10
def prune(node):
    if len(node['children']) > 10:
        node['children'].sort(key=lambda x: x['count'], reverse=True)
        node['children'] = node['children'][:10]
    for child in node['children']:
        prune(child)

prune(root)
print(f'Max depth AFTER pruning: {calc_depth(root)}')

# Count nodes after pruning
def count_nodes(node):
    return 1 + sum(count_nodes(child) for child in node['children'])
print(f'Nodes after pruning: {count_nodes(root):,}')
