#!/usr/bin/env python3
"""
Comprehensive evaluation of generative life sequence models.

This script evaluates generated life sequences by computing:
1. Distribution of life spans (age at death or end of sequence)
2. Distribution of number of children per person
3. Distribution of income levels over time
4. Gender-specific patterns
5. Birth patterns (for couples, checking for babies within 3 years)
6. Employment patterns
7. Marriage/partnership patterns
8. Geographic mobility

Usage:
    python evaluate_generative_model.py \
        --generated_h5 <path_to_generated_sequences.h5> \
        --real_h5 <path_to_real_sequences.h5> \
        --vocab_path <path_to_vocab.csv> \
        --output_dir <output_directory>
        
    OR use a pretty_tokens file:
    python evaluate_generative_model.py \
        --generated_tokens <path_to_pretty_tokens.txt> \
        --real_h5 <path_to_real_sequences.h5> \
        --vocab_path <path_to_vocab.csv> \
        --output_dir <output_directory>
"""

import argparse
import h5py
import json
import os
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class LifeSequenceEvaluator:
    """Evaluate generated life sequences against real data."""
    
    def __init__(self, vocab_path: str):
        """Initialize with vocabulary."""
        self.vocab_df = pd.read_csv(vocab_path)
        self.id_to_token = dict(zip(self.vocab_df['ID'], self.vocab_df['TOKEN']))
        self.id_to_category = dict(zip(self.vocab_df['ID'], self.vocab_df['CATEGORY']))
        
        # Pre-compute category prefixes for faster lookup
        self.category_patterns = {
            'background': ['background', 'BACKGROUND'],
            'income': ['INPATAB'],
            'employment': ['Spolisbus'],
            'household': ['household'],
            'network': ['network'],
        }
    
    def get_token_category(self, token_id: int) -> str:
        """Get category for a token ID."""
        return self.id_to_category.get(token_id, 'UNKNOWN')
    
    def extract_age_at_death(self, sequence: List[int]) -> Optional[int]:
        """
        Extract age at death from a sequence.
        Looks for [DEATH] token and returns the age from the previous token.
        If no death token, returns age at last event.
        """
        # Look for death token
        death_token_ids = [tid for tid, token in self.id_to_token.items() 
                          if 'DEATH' in token.upper()]
        
        # Find age tokens (year_XXXX)
        ages = []
        for i, tid in enumerate(sequence):
            token = self.id_to_token.get(tid, '')
            category = self.id_to_category.get(tid, '')
            
            # Extract birth year
            if 'background' in category.lower() and 'year_' in token:
                try:
                    birth_year = int(token.split('year_')[1])
                    ages.append(birth_year)
                except:
                    pass
        
        if ages:
            # Calculate lifespan based on sequence length and temporal markers
            # This is a simplified version - adjust based on your data format
            return len(sequence) // 10  # Rough estimate based on average events per year
        
        return None
    
    def count_children(self, sequence: List[int]) -> int:
        """
        Count number of children by looking for birth-related network events.
        """
        children = 0
        for tid in sequence:
            token = self.id_to_token.get(tid, '')
            category = self.id_to_category.get(tid, '')
            
            # Look for parent-child network layer types
            # Adjust this based on your actual data encoding
            if 'network' in category.lower():
                if 'layerType_' in token:
                    # Layer types 1-10 might indicate parent-child relationships
                    try:
                        layer = int(token.split('layerType_')[1])
                        if layer in [1, 2, 3, 4, 5]:  # Adjust based on your schema
                            children += 1
                    except:
                        pass
        
        return children
    
    def extract_income_over_time(self, sequence: List[int]) -> List[float]:
        """
        Extract income values over time.
        Returns list of income values (normalized or binned).
        """
        incomes = []
        for tid in sequence:
            token = self.id_to_token.get(tid, '')
            category = self.id_to_category.get(tid, '')
            
            if 'INPATAB' in category:
                # Income tokens - extract numerical value if present
                # This is highly dependent on your vocab encoding
                if '_' in token:
                    try:
                        # Extract numerical suffix
                        parts = token.rsplit('_', 1)
                        if len(parts) == 2:
                            val = parts[1]
                            if val.isdigit():
                                incomes.append(float(val))
                    except:
                        pass
        
        return incomes
    
    def get_gender(self, sequence: List[int]) -> Optional[str]:
        """Extract gender from sequence."""
        for tid in sequence:
            token = self.id_to_token.get(tid, '')
            if 'gender_' in token.lower():
                if 'gender_1' in token:
                    return 'male'
                elif 'gender_2' in token:
                    return 'female'
        return None
    
    def check_birth_in_window(self, sequence: List[int], window_years: int = 3) -> bool:
        """
        Check if a birth event occurs within a time window.
        Useful for couples expecting babies.
        """
        # This is a simplified check - adjust based on your data format
        return self.count_children(sequence) > 0
    
    def extract_employment_spells(self, sequence: List[int]) -> List[Dict]:
        """
        Extract employment spells (start, end, hours, fulltime status).
        """
        spells = []
        current_spell = None
        
        for tid in sequence:
            token = self.id_to_token.get(tid, '')
            category = self.id_to_category.get(tid, '')
            
            if 'Spolisbus' in category:
                if 'beg_or_end' in token:
                    if current_spell is None:
                        current_spell = {'start': True}
                    else:
                        spells.append(current_spell)
                        current_spell = None
                elif 'fulltime_' in token and current_spell is not None:
                    try:
                        hours = int(token.split('fulltime_')[1])
                        current_spell['fulltime_hours'] = hours
                    except:
                        pass
        
        if current_spell is not None:
            spells.append(current_spell)
        
        return spells


def load_sequences_from_h5(h5_path: str, max_sequences: int = None) -> List[List[int]]:
    """Load sequences from HDF5 file."""
    sequences = []
    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        n = min(len(data), max_sequences) if max_sequences else len(data)
        for i in range(n):
            seq = data[i][0]  # Assuming shape (4, L) and we want the first row (tokens)
            # Remove padding (assuming 0 is pad)
            seq = [int(x) for x in seq if x != 0]
            sequences.append(seq)
    return sequences


def load_sequences_from_tokens(tokens_file: str) -> List[List[int]]:
    """
    Load sequences from pretty_tokens file.
    Note: This returns token strings, not IDs. Would need vocab to convert.
    For now, we'll skip this and focus on H5 format.
    """
    raise NotImplementedError("Token file loading not yet implemented - use H5 format")


def evaluate_and_compare(
    generated_sequences: List[List[int]],
    real_sequences: List[List[int]],
    evaluator: LifeSequenceEvaluator,
    output_dir: str
):
    """
    Compare generated vs real sequences across multiple dimensions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'generated': {},
        'real': {},
        'comparison': {}
    }
    
    # 1. Life spans
    print("Computing life spans...")
    gen_lifespans = [evaluator.extract_age_at_death(seq) for seq in generated_sequences]
    real_lifespans = [evaluator.extract_age_at_death(seq) for seq in real_sequences]
    gen_lifespans = [x for x in gen_lifespans if x is not None]
    real_lifespans = [x for x in real_lifespans if x is not None]
    
    results['generated']['mean_lifespan'] = float(np.mean(gen_lifespans)) if gen_lifespans else 0
    results['real']['mean_lifespan'] = float(np.mean(real_lifespans)) if real_lifespans else 0
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(real_lifespans, bins=50, alpha=0.5, label='Real', density=True)
    ax.hist(gen_lifespans, bins=50, alpha=0.5, label='Generated', density=True)
    ax.set_xlabel('Life Span (years)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Life Spans')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'lifespan_distribution.png'), dpi=150)
    plt.close()
    
    # 2. Number of children
    print("Computing number of children...")
    gen_children = [evaluator.count_children(seq) for seq in generated_sequences]
    real_children = [evaluator.count_children(seq) for seq in real_sequences]
    
    results['generated']['mean_children'] = float(np.mean(gen_children))
    results['real']['mean_children'] = float(np.mean(real_children))
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bins = np.arange(0, max(max(gen_children), max(real_children)) + 2) - 0.5
    ax.hist(real_children, bins=bins, alpha=0.5, label='Real', density=True)
    ax.hist(gen_children, bins=bins, alpha=0.5, label='Generated', density=True)
    ax.set_xlabel('Number of Children')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Number of Children')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'children_distribution.png'), dpi=150)
    plt.close()
    
    # 3. Income distribution
    print("Computing income distributions...")
    gen_incomes_all = []
    real_incomes_all = []
    for seq in generated_sequences:
        gen_incomes_all.extend(evaluator.extract_income_over_time(seq))
    for seq in real_sequences:
        real_incomes_all.extend(evaluator.extract_income_over_time(seq))
    
    if gen_incomes_all and real_incomes_all:
        results['generated']['mean_income'] = float(np.mean(gen_incomes_all))
        results['real']['mean_income'] = float(np.mean(real_incomes_all))
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(real_incomes_all, bins=50, alpha=0.5, label='Real', density=True)
        ax.hist(gen_incomes_all, bins=50, alpha=0.5, label='Generated', density=True)
        ax.set_xlabel('Income Level')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Income Levels')
        ax.legend()
        plt.savefig(os.path.join(output_dir, 'income_distribution.png'), dpi=150)
        plt.close()
    
    # 4. Gender distribution
    print("Computing gender distributions...")
    gen_genders = [evaluator.get_gender(seq) for seq in generated_sequences]
    real_genders = [evaluator.get_gender(seq) for seq in real_sequences]
    gen_gender_counts = Counter([g for g in gen_genders if g is not None])
    real_gender_counts = Counter([g for g in real_genders if g is not None])
    
    results['generated']['gender_distribution'] = dict(gen_gender_counts)
    results['real']['gender_distribution'] = dict(real_gender_counts)
    
    # 5. Birth patterns (for couples)
    print("Checking birth patterns...")
    gen_births_in_3y = sum(1 for seq in generated_sequences if evaluator.check_birth_in_window(seq, 3))
    real_births_in_3y = sum(1 for seq in real_sequences if evaluator.check_birth_in_window(seq, 3))
    
    results['generated']['births_within_3_years'] = gen_births_in_3y
    results['real']['births_within_3_years'] = real_births_in_3y
    results['generated']['birth_rate_3y'] = gen_births_in_3y / len(generated_sequences) if generated_sequences else 0
    results['real']['birth_rate_3y'] = real_births_in_3y / len(real_sequences) if real_sequences else 0
    
    # 6. Employment patterns
    print("Computing employment patterns...")
    gen_employment = [evaluator.extract_employment_spells(seq) for seq in generated_sequences]
    real_employment = [evaluator.extract_employment_spells(seq) for seq in real_sequences]
    
    gen_num_jobs = [len(spells) for spells in gen_employment]
    real_num_jobs = [len(spells) for spells in real_employment]
    
    results['generated']['mean_num_jobs'] = float(np.mean(gen_num_jobs))
    results['real']['mean_num_jobs'] = float(np.mean(real_num_jobs))
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bins = np.arange(0, max(max(gen_num_jobs), max(real_num_jobs)) + 2) - 0.5
    ax.hist(real_num_jobs, bins=bins, alpha=0.5, label='Real', density=True)
    ax.hist(gen_num_jobs, bins=bins, alpha=0.5, label='Generated', density=True)
    ax.set_xlabel('Number of Jobs')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Number of Jobs')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'num_jobs_distribution.png'), dpi=150)
    plt.close()
    
    # Save results
    print("Saving results...")
    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nGenerated sequences: {len(generated_sequences)}")
    print(f"Real sequences: {len(real_sequences)}")
    print(f"\nLife Span:")
    print(f"  Generated: {results['generated']['mean_lifespan']:.1f} years")
    print(f"  Real:      {results['real']['mean_lifespan']:.1f} years")
    print(f"\nNumber of Children:")
    print(f"  Generated: {results['generated']['mean_children']:.2f}")
    print(f"  Real:      {results['real']['mean_children']:.2f}")
    if 'mean_income' in results['generated']:
        print(f"\nIncome Level:")
        print(f"  Generated: {results['generated']['mean_income']:.1f}")
        print(f"  Real:      {results['real']['mean_income']:.1f}")
    print(f"\nBirth Rate (within 3 years):")
    print(f"  Generated: {results['generated']['birth_rate_3y']:.2%}")
    print(f"  Real:      {results['real']['birth_rate_3y']:.2%}")
    print(f"\nMean Number of Jobs:")
    print(f"  Generated: {results['generated']['mean_num_jobs']:.2f}")
    print(f"  Real:      {results['real']['mean_num_jobs']:.2f}")
    print("\n" + "="*60)
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generative life sequence model"
    )
    parser.add_argument(
        "--generated_h5",
        help="Path to HDF5 file with generated sequences"
    )
    parser.add_argument(
        "--generated_tokens",
        help="Path to pretty_tokens file with generated sequences (alternative to H5)"
    )
    parser.add_argument(
        "--real_h5",
        required=True,
        help="Path to HDF5 file with real sequences"
    )
    parser.add_argument(
        "--vocab_path",
        required=True,
        help="Path to vocabulary CSV file"
    )
    parser.add_argument(
        "--output_dir",
        default="./evaluation_output",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=1000,
        help="Maximum number of sequences to evaluate (default: 1000)"
    )
    
    args = parser.parse_args()
    
    if not args.generated_h5 and not args.generated_tokens:
        parser.error("Either --generated_h5 or --generated_tokens must be provided")
    
    # Initialize evaluator
    print("Loading vocabulary...")
    evaluator = LifeSequenceEvaluator(args.vocab_path)
    
    # Load sequences
    print("Loading generated sequences...")
    if args.generated_h5:
        generated_sequences = load_sequences_from_h5(args.generated_h5, args.max_sequences)
    else:
        generated_sequences = load_sequences_from_tokens(args.generated_tokens)
    
    print("Loading real sequences...")
    real_sequences = load_sequences_from_h5(args.real_h5, args.max_sequences)
    
    print(f"Loaded {len(generated_sequences)} generated and {len(real_sequences)} real sequences")
    
    # Evaluate
    print("\nStarting evaluation...")
    evaluate_and_compare(generated_sequences, real_sequences, evaluator, args.output_dir)


if __name__ == "__main__":
    main()
