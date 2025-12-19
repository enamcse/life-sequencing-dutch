#!/usr/bin/env python3
"""
Analyze generative model outputs and compute statistics.

This script analyzes the output from generative_infer.py and computes:
1. Category distribution of generated tokens
2. Token match rate (how many generated tokens match the original)
3. Category-specific match rates
4. Diversity metrics (unique tokens, repetition rates)
5. Per-sequence statistics

Usage:
    python analyze_generative_output.py <input_file> [--output_dir <dir>]
    
Example:
    python analyze_generative_output.py \
        /projects/0/prjs1589/stonybrook/llm/gen_out/pretty_tokens_20251212_repaired.txt \
        --output_dir /projects/0/prjs1589/stonybrook/llm/gen_out/analysis/
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def parse_line(line: str) -> Tuple[str, int, List[Tuple[str, str]]]:
    """
    Parse a line from the output file.
    
    Returns:
        (line_type, sequence_num, token_category_pairs)
        where each token_category_pair is (token, category) if with_category else (token, None)
    """
    parts = line.strip().split(',', 2)
    if len(parts) < 3:
        return None, None, None
    
    # Extract type and sequence number - now supporting PREFIX, GROUND TRUTH, and GENERATED
    match = re.match(r'(ORIGINAL PREFIX|GROUND TRUTH CONTINUATION|GENERATED).*\(Sequence (\d+)\)', parts[0])
    if not match:
        return None, None, None
    
    line_type = match.group(1)
    seq_num = int(match.group(2))
    
    # Parse tokens
    tokens_str = parts[2]
    token_parts = tokens_str.split(',')
    
    tokens = []
    for part in token_parts:
        if '|' in part:  # with_category format
            token, category = part.split('|', 1)
            tokens.append((token, category))
        else:
            tokens.append((part, None))
    
    return line_type, seq_num, tokens


def load_sequences(input_file: str) -> Dict[int, Dict[str, List[Tuple[str, str]]]]:
    """
    Load sequences from file.
    
    Returns:
        {sequence_num: {'ORIGINAL PREFIX': [...], 'GROUND TRUTH CONTINUATION': [...], 'GENERATED': [...]}}
    """
    sequences = defaultdict(dict)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_type, seq_num, tokens = parse_line(line)
            if line_type is None:
                continue
            sequences[seq_num][line_type] = tokens
    
    return sequences


def compute_category_distribution(tokens: List[Tuple[str, str]]) -> Counter:
    """Count tokens by category."""
    return Counter(cat for token, cat in tokens if cat is not None)


def compute_token_match_rate(original: List[Tuple[str, str]], generated: List[Tuple[str, str]]) -> float:
    """
    Compute how many generated tokens match the original continuation.
    Note: This is a simple exact match at the same position.
    """
    matches = sum(1 for o, g in zip(original, generated) if o[0] == g[0])
    total = max(len(original), len(generated))
    return matches / total if total > 0 else 0.0


def compute_category_match_rate(original: List[Tuple[str, str]], generated: List[Tuple[str, str]]) -> float:
    """Compute how many generated tokens have the correct category."""
    matches = sum(1 for o, g in zip(original, generated) if o[1] == g[1])
    total = max(len(original), len(generated))
    return matches / total if total > 0 else 0.0


def compute_diversity_metrics(tokens: List[Tuple[str, str]]) -> Dict:
    """
    Compute diversity metrics for a sequence.
    """
    token_list = [t[0] for t in tokens]
    
    unique_tokens = len(set(token_list))
    total_tokens = len(token_list)
    
    # Repetition: count consecutive duplicates
    consecutive_repeats = sum(1 for i in range(1, len(token_list)) if token_list[i] == token_list[i-1])
    
    # Most common token
    token_counts = Counter(token_list)
    most_common_token, most_common_count = token_counts.most_common(1)[0] if token_counts else (None, 0)
    
    return {
        'unique_tokens': unique_tokens,
        'total_tokens': total_tokens,
        'unique_ratio': unique_tokens / total_tokens if total_tokens > 0 else 0,
        'consecutive_repeats': consecutive_repeats,
        'repeat_ratio': consecutive_repeats / (total_tokens - 1) if total_tokens > 1 else 0,
        'most_common_token': most_common_token,
        'most_common_count': most_common_count,
        'most_common_ratio': most_common_count / total_tokens if total_tokens > 0 else 0,
    }


def analyze_sequences(sequences: Dict[int, Dict[str, List[Tuple[str, str]]]]) -> Dict:
    """
    Perform comprehensive analysis on all sequences.
    Now compares GENERATED vs GROUND TRUTH CONTINUATION (not prefix).
    """
    results = {
        'num_sequences': len(sequences),
        'generated_category_dist': Counter(),
        'ground_truth_category_dist': Counter(),
        'token_match_rates': [],
        'category_match_rates': [],
        'per_sequence_stats': [],
        'diversity_stats': [],
        'category_specific_matches': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'sequences_with_ground_truth': 0,
        'sequences_without_ground_truth': 0,
    }
    
    for seq_num in sorted(sequences.keys()):
        seq_data = sequences[seq_num]
        
        if 'GENERATED' not in seq_data:
            continue
        
        generated = seq_data['GENERATED']
        
        # Check if we have ground truth continuation (not just prefix)
        ground_truth = seq_data.get('GROUND TRUTH CONTINUATION', None)
        
        if ground_truth is None:
            # Fallback: warn user and skip match rate calculations
            results['sequences_without_ground_truth'] += 1
            # Still compute diversity on generated
            diversity = compute_diversity_metrics(generated)
            results['diversity_stats'].append(diversity)
            results['generated_category_dist'].update(compute_category_distribution(generated))
            continue
        
        results['sequences_with_ground_truth'] += 1
        
        # Category distribution
        results['generated_category_dist'].update(compute_category_distribution(generated))
        results['ground_truth_category_dist'].update(compute_category_distribution(ground_truth))
        
        # Match rates (GENERATED vs GROUND TRUTH)
        token_match = compute_token_match_rate(ground_truth, generated)
        category_match = compute_category_match_rate(ground_truth, generated)
        results['token_match_rates'].append(token_match)
        results['category_match_rates'].append(category_match)
        
        # Category-specific matches
        for (gt_token, gt_cat), (g_token, g_cat) in zip(ground_truth, generated):
            if gt_cat is not None:
                results['category_specific_matches'][gt_cat]['total'] += 1
                if gt_token == g_token:
                    results['category_specific_matches'][gt_cat]['correct'] += 1
        
        # Diversity
        diversity = compute_diversity_metrics(generated)
        results['diversity_stats'].append(diversity)
        
        # Per-sequence stats
        results['per_sequence_stats'].append({
            'sequence_num': seq_num,
            'ground_truth_len': len(ground_truth),
            'generated_len': len(generated),
            'token_match_rate': token_match,
            'category_match_rate': category_match,
            'unique_ratio': diversity['unique_ratio'],
            'repeat_ratio': diversity['repeat_ratio'],
        })
    
    return results


def plot_category_distribution(category_dist: Counter, title: str, output_path: str):
    """Plot category distribution as a bar chart."""
    if not category_dist:
        print(f"Warning: No data to plot for {title}")
        return
    
    # Get top 30 categories
    top_cats = category_dist.most_common(30)
    cats, counts = zip(*top_cats)
    
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(cats)), counts)
    plt.xticks(range(len(cats)), cats, rotation=90, ha='right')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_match_rates(token_matches: List[float], category_matches: List[float], output_path: str):
    """Plot distribution of match rates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(token_matches, bins=20, edgecolor='black')
    axes[0].set_xlabel('Token Match Rate')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Token Match Rate Distribution\n(Mean: {np.mean(token_matches):.3f})')
    axes[0].axvline(np.mean(token_matches), color='red', linestyle='--', label='Mean')
    axes[0].legend()
    
    axes[1].hist(category_matches, bins=20, edgecolor='black')
    axes[1].set_xlabel('Category Match Rate')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Category Match Rate Distribution\n(Mean: {np.mean(category_matches):.3f})')
    axes[1].axvline(np.mean(category_matches), color='red', linestyle='--', label='Mean')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_diversity_metrics(diversity_stats: List[Dict], output_path: str):
    """Plot diversity metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    unique_ratios = [d['unique_ratio'] for d in diversity_stats]
    repeat_ratios = [d['repeat_ratio'] for d in diversity_stats]
    most_common_ratios = [d['most_common_ratio'] for d in diversity_stats]
    
    axes[0, 0].hist(unique_ratios, bins=20, edgecolor='black')
    axes[0, 0].set_xlabel('Unique Token Ratio')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Unique Token Ratio\n(Mean: {np.mean(unique_ratios):.3f})')
    axes[0, 0].axvline(np.mean(unique_ratios), color='red', linestyle='--')
    
    axes[0, 1].hist(repeat_ratios, bins=20, edgecolor='black')
    axes[0, 1].set_xlabel('Consecutive Repeat Ratio')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Consecutive Repeat Ratio\n(Mean: {np.mean(repeat_ratios):.3f})')
    axes[0, 1].axvline(np.mean(repeat_ratios), color='red', linestyle='--')
    
    axes[1, 0].hist(most_common_ratios, bins=20, edgecolor='black')
    axes[1, 0].set_xlabel('Most Common Token Ratio')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Most Common Token Ratio\n(Mean: {np.mean(most_common_ratios):.3f})')
    axes[1, 0].axvline(np.mean(most_common_ratios), color='red', linestyle='--')
    
    # Scatter: unique ratio vs repeat ratio
    axes[1, 1].scatter(unique_ratios, repeat_ratios, alpha=0.5)
    axes[1, 1].set_xlabel('Unique Token Ratio')
    axes[1, 1].set_ylabel('Consecutive Repeat Ratio')
    axes[1, 1].set_title('Diversity vs Repetition')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_summary(results: Dict, output_path: str):
    """Save summary statistics to JSON."""
    summary = {
        'num_sequences': results['num_sequences'],
        'sequences_with_ground_truth': results['sequences_with_ground_truth'],
        'sequences_without_ground_truth': results['sequences_without_ground_truth'],
        'mean_token_match_rate': float(np.mean(results['token_match_rates'])) if results['token_match_rates'] else 0,
        'std_token_match_rate': float(np.std(results['token_match_rates'])) if results['token_match_rates'] else 0,
        'mean_category_match_rate': float(np.mean(results['category_match_rates'])) if results['category_match_rates'] else 0,
        'std_category_match_rate': float(np.std(results['category_match_rates'])) if results['category_match_rates'] else 0,
        'top_20_generated_categories': dict(results['generated_category_dist'].most_common(20)),
        'top_20_ground_truth_categories': dict(results['ground_truth_category_dist'].most_common(20)),
        'category_specific_match_rates': {
            cat: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            for cat, stats in results['category_specific_matches'].items()
        },
        'diversity_stats': {
            'mean_unique_ratio': float(np.mean([d['unique_ratio'] for d in results['diversity_stats']])) if results['diversity_stats'] else 0,
            'mean_repeat_ratio': float(np.mean([d['repeat_ratio'] for d in results['diversity_stats']])) if results['diversity_stats'] else 0,
            'mean_most_common_ratio': float(np.mean([d['most_common_ratio'] for d in results['diversity_stats']])) if results['diversity_stats'] else 0,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


def save_per_sequence_stats(per_seq_stats: List[Dict], output_path: str):
    """Save per-sequence statistics to CSV."""
    df = pd.DataFrame(per_seq_stats)
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze generative model outputs"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input file (repaired output from generative_infer.py)"
    )
    parser.add_argument(
        "--output_dir",
        default="./analysis_output",
        help="Directory to save analysis results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading sequences from: {args.input_file}")
    sequences = load_sequences(args.input_file)
    print(f"Loaded {len(sequences)} sequences")
    
    print("Analyzing sequences...")
    results = analyze_sequences(sequences)
    
    print("Generating plots...")
    plot_category_distribution(
        results['generated_category_dist'],
        'Generated Token Category Distribution (Top 30)',
        os.path.join(args.output_dir, 'generated_category_dist.png')
    )
    
    plot_category_distribution(
        results['ground_truth_category_dist'],
        'Ground Truth Continuation Category Distribution (Top 30)',
        os.path.join(args.output_dir, 'ground_truth_category_dist.png')
    )
    
    if results['token_match_rates']:
        plot_match_rates(
            results['token_match_rates'],
            results['category_match_rates'],
            os.path.join(args.output_dir, 'match_rates.png')
        )
    
    if results['diversity_stats']:
        plot_diversity_metrics(
            results['diversity_stats'],
            os.path.join(args.output_dir, 'diversity_metrics.png')
        )
    
    print("Saving results...")
    save_summary(results, os.path.join(args.output_dir, 'summary.json'))
    save_per_sequence_stats(
        results['per_sequence_stats'],
        os.path.join(args.output_dir, 'per_sequence_stats.csv')
    )
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")
    print(f"\nQuick Summary:")
    print(f"  - Sequences analyzed: {results['num_sequences']}")
    print(f"  - Sequences with ground truth: {results['sequences_with_ground_truth']}")
    print(f"  - Sequences without ground truth: {results['sequences_without_ground_truth']}")
    if results['token_match_rates']:
        print(f"  - Mean token match rate: {np.mean(results['token_match_rates']):.3f}")
        print(f"  - Mean category match rate: {np.mean(results['category_match_rates']):.3f}")
    if results['diversity_stats']:
        print(f"  - Mean unique token ratio: {np.mean([d['unique_ratio'] for d in results['diversity_stats']]):.3f}")
        print(f"  - Mean repeat ratio: {np.mean([d['repeat_ratio'] for d in results['diversity_stats']]):.3f}")
    if results['sequences_without_ground_truth'] > 0:
        print(f"\n⚠️  WARNING: {results['sequences_without_ground_truth']} sequences have no ground truth continuation.")
        print(f"    Match rates are only computed for sequences with ground truth.")


if __name__ == "__main__":
    main()
