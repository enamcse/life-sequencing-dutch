#!/usr/bin/env python3
"""
Statistical Evaluation of Generative Model

This script performs comprehensive statistical evaluation of generative models by:
1. Testing on multiple real people's sequences
2. Generating multiple times per person (for statistical significance)
3. Varying prefix lengths progressively
4. Measuring both ordered and unordered matches
5. Computing statistics by token and higher category
6. Supporting single or multiple generation horizons

Usage:
    python statistical_evaluation.py --hparams path/to/eval_hparams.txt

Output:
    - Multiple CSV files with statistics
    - Registry file mapping run IDs to parameters
    - Summary plots and reports
"""

import argparse
import csv
import h5py
import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax

# Project imports
from pop2vec.llm.src.new_code.utils import (
    read_hparams,
    get_vocab_size,
    load_special_ids,
    load_vocab_df,
)
from pop2vec.llm.src.new_code.load_data import CustomLazyHDF5Dataset
from pop2vec.llm.src.transformer.models import TransformerEncoder

# Logging setup
logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for statistical evaluation."""
    # Paths
    mlm_path: str
    vocab_path: str
    pretrained_model_path: str
    output_dir: str
    
    # Evaluation scope
    num_people: int = 10  # Number of real people to evaluate
    generations_per_person: int = 100  # Number of generations per person
    
    # Prefix progression
    prefix_start: int = 1  # Always start with 1 token ([CLS])
    prefix_jump: int = 100  # Jump size for prefix lengths (1, 101, 201, 301, ...)
    
    # Generation horizons
    horizon_mode: str = "single"  # "single" or "multiple"
    horizon_single: int = 20  # For single mode
    horizon_gap: int = 20  # For multiple mode: 20, 40, 60, ...
    horizon_max: int = 200  # Max horizon for multiple mode
    
    # Matching modes
    compute_ordered: bool = True  # Compute ordered (position-based) matching
    compute_unordered: bool = True  # Compute unordered (set-based) matching
    
    # Categorization
    use_higher_category: bool = True  # Use HIGHER_CATEGORY from vocab
    
    # Sampling
    top_k: int = 20
    temperature: float = 1.0
    
    # Special tokens
    pad_token: str = "[PAD]"
    cls_token: str = "[CLS]"
    death_token: str = "[DEATH]"
    
    # Output control
    save_generated_sequences: bool = False  # Save actual generated sequences (large!)
    registry_mode: str = "subfolder"  # "subfolder" or "registry"


class StatisticalEvaluator:
    """Main evaluator class for statistical analysis."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Load vocabulary
        logger.info(f"Loading vocabulary from: {config.vocab_path}")
        self.vocab_df = load_vocab_df(config.vocab_path)
        
        # Check if HIGHER_CATEGORY exists
        if config.use_higher_category:
            if "HIGHER_CATEGORY" not in self.vocab_df.columns:
                logger.warning("HIGHER_CATEGORY not found in vocab, will use CATEGORY")
                config.use_higher_category = False
            else:
                logger.info("Using HIGHER_CATEGORY for aggregated statistics")
        
        # Load special token IDs
        specials = load_special_ids(
            config.vocab_path,
            pad_token=config.pad_token,
            cls_token=config.cls_token,
            death_token=config.death_token,
        )
        self.pad_id = specials["pad_id"]
        self.cls_id = specials["cls_id"]
        self.death_id = specials["death_id"]
        
        # Create token ID to category/higher_category mappings
        self.id_to_category = dict(zip(self.vocab_df['ID'], self.vocab_df['CATEGORY']))
        if config.use_higher_category:
            self.id_to_higher_category = dict(zip(
                self.vocab_df['ID'],
                self.vocab_df['HIGHER_CATEGORY']
            ))
        
        # Load model
        logger.info(f"Loading model from: {config.pretrained_model_path}")
        self.model = TransformerEncoder.load_from_checkpoint(
            config.pretrained_model_path,
            strict=False
        )
        self.model.eval().to(self.device)
        
        # Load dataset
        logger.info(f"Loading dataset from: {config.mlm_path}")
        self.dataset = CustomLazyHDF5Dataset(
            config.mlm_path,
            validation=False,
            num_val_items=100000,
            mlm_encoded=False,
            inference=True,
        )
        
        # Generate run ID and setup output structure
        self.run_id = self._generate_run_id()
        self._setup_output_structure()
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID based on timestamp."""
        return f"run_{int(time.time())}"
    
    def _setup_output_structure(self):
        """Setup output directory structure."""
        if self.config.registry_mode == "subfolder":
            # Create subfolder for this run
            self.run_output_dir = os.path.join(self.config.output_dir, self.run_id)
            os.makedirs(self.run_output_dir, exist_ok=True)
            
            # Save config
            config_path = os.path.join(self.run_output_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            
            logger.info(f"Created run directory: {self.run_output_dir}")
        else:
            # Use flat structure with registry
            self.run_output_dir = self.config.output_dir
            
            # Update registry
            registry_path = os.path.join(self.config.output_dir, "registry.csv")
            registry_exists = os.path.exists(registry_path)
            
            with open(registry_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not registry_exists:
                    # Write header
                    writer.writerow(['run_id', 'timestamp', 'config_json'])
                writer.writerow([
                    self.run_id,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    json.dumps(asdict(self.config))
                ])
            
            logger.info(f"Registered run in: {registry_path}")
    
    def _get_horizons(self) -> List[int]:
        """Get list of horizons to evaluate."""
        if self.config.horizon_mode == "single":
            return [self.config.horizon_single]
        else:  # multiple
            horizons = []
            h = self.config.horizon_gap
            while h <= self.config.horizon_max:
                horizons.append(h)
                h += self.config.horizon_gap
            return horizons
    
    def _get_prefix_lengths(self, max_seq_len: int, horizon: int) -> List[int]:
        """Get list of prefix lengths to evaluate for a given sequence and horizon."""
        prefix_lengths = [self.config.prefix_start]  # Always start with 1
        
        current = self.config.prefix_start + self.config.prefix_jump
        while current + horizon <= max_seq_len:
            prefix_lengths.append(current)
            current += self.config.prefix_jump
        
        return prefix_lengths
    
    @torch.no_grad()
    def _generate_tokens(
        self,
        prefix_4stream: torch.Tensor,
        pad_mask: torch.Tensor,
        horizon: int
    ) -> List[int]:
        """Generate tokens given a prefix."""
        x = prefix_4stream.unsqueeze(0).to(self.device)  # (1, 4, L)
        pm = pad_mask.unsqueeze(0).to(self.device)  # (1, L)
        out_tokens = []
        
        for _ in range(horizon):
            logits = self.model({"input_ids": x, "padding_mask": pm})
            last_logits = logits[:, -1, :] / max(1e-8, self.config.temperature)
            
            if self.config.top_k > 0:
                vals, idxs = torch.topk(last_logits, k=self.config.top_k, dim=-1)
                probs = softmax(vals, dim=-1)
                next_token = idxs.gather(-1, torch.multinomial(probs, 1)).squeeze(-1)
            else:
                next_token = torch.argmax(last_logits, dim=-1)
            
            tid = int(next_token.item())
            out_tokens.append(tid)
            
            # Stop if death token
            if self.death_id is not None and tid == self.death_id:
                break
            
            # Append token to sequence
            last_age = x[0, 1, -1].item()
            last_day = x[0, 2, -1].item()
            new_step = torch.tensor(
                [[tid], [last_age], [last_day], [1]],
                dtype=torch.long,
                device=self.device
            )
            x = torch.cat([x, new_step.unsqueeze(0)], dim=2)
            pm = torch.cat([pm, torch.ones(1, 1, dtype=pm.dtype, device=self.device)], dim=1)
        
        return out_tokens
    
    def _compute_ordered_match(
        self,
        ground_truth: List[int],
        generated: List[int]
    ) -> Dict[str, float]:
        """Compute ordered (position-based) matching statistics."""
        min_len = min(len(ground_truth), len(generated))
        if min_len == 0:
            return {'exact_matches': 0, 'match_rate': 0.0}
        
        exact_matches = sum(
            1 for i in range(min_len) if ground_truth[i] == generated[i]
        )
        
        return {
            'exact_matches': exact_matches,
            'match_rate': exact_matches / min_len
        }
    
    def _compute_unordered_match(
        self,
        ground_truth: List[int],
        generated: List[int]
    ) -> Dict[str, float]:
        """Compute unordered (set-based) matching statistics."""
        gt_set = set(ground_truth)
        gen_set = set(generated)
        
        if len(gt_set) == 0 and len(gen_set) == 0:
            return {
                'common_tokens': 0,
                'jaccard_similarity': 1.0,
                'precision': 1.0,
                'recall': 1.0
            }
        
        common = gt_set.intersection(gen_set)
        union = gt_set.union(gen_set)
        
        jaccard = len(common) / len(union) if len(union) > 0 else 0.0
        precision = len(common) / len(gen_set) if len(gen_set) > 0 else 0.0
        recall = len(common) / len(gt_set) if len(gt_set) > 0 else 0.0
        
        return {
            'common_tokens': len(common),
            'jaccard_similarity': jaccard,
            'precision': precision,
            'recall': recall
        }
    
    def _compute_category_statistics(
        self,
        ground_truth: List[int],
        generated: List[int],
        use_higher: bool = False
    ) -> Dict[str, any]:
        """Compute category-level statistics."""
        id_to_cat = self.id_to_higher_category if use_higher else self.id_to_category
        
        gt_cats = [id_to_cat.get(tid, "UNKNOWN") for tid in ground_truth]
        gen_cats = [id_to_cat.get(tid, "UNKNOWN") for tid in generated]
        
        # Category distributions
        gt_dist = Counter(gt_cats)
        gen_dist = Counter(gen_cats)
        
        # Ordered category match
        min_len = min(len(gt_cats), len(gen_cats))
        cat_exact_matches = sum(
            1 for i in range(min_len) if gt_cats[i] == gen_cats[i]
        ) if min_len > 0 else 0
        
        cat_match_rate = cat_exact_matches / min_len if min_len > 0 else 0.0
        
        # Unordered category match
        gt_cat_set = set(gt_cats)
        gen_cat_set = set(gen_cats)
        common_cats = gt_cat_set.intersection(gen_cat_set)
        
        return {
            'category_match_rate': cat_match_rate,
            'common_categories': len(common_cats),
            'gt_category_dist': dict(gt_dist),
            'gen_category_dist': dict(gen_dist)
        }
    
    def _evaluate_single_generation(
        self,
        person_idx: int,
        generation_num: int,
        prefix_len: int,
        horizon: int,
        full_sequence: torch.Tensor,
        full_mask: torch.Tensor
    ) -> Dict:
        """Evaluate a single generation."""
        # Extract prefix and ground truth
        prefix_4stream = full_sequence[:, :prefix_len]
        prefix_mask = full_mask[:prefix_len]
        
        ground_truth_end = min(prefix_len + horizon, full_sequence.size(1))
        ground_truth = full_sequence[0, prefix_len:ground_truth_end].tolist()
        
        # Generate
        generated = self._generate_tokens(prefix_4stream, prefix_mask, horizon)
        
        # Compute statistics
        stats = {
            'person_idx': person_idx,
            'generation_num': generation_num,
            'prefix_len': prefix_len,
            'horizon': horizon,
            'gt_len': len(ground_truth),
            'gen_len': len(generated),
        }
        
        # Ordered matching
        if self.config.compute_ordered:
            ordered_stats = self._compute_ordered_match(ground_truth, generated)
            stats.update({f'ordered_{k}': v for k, v in ordered_stats.items()})
        
        # Unordered matching
        if self.config.compute_unordered:
            unordered_stats = self._compute_unordered_match(ground_truth, generated)
            stats.update({f'unordered_{k}': v for k, v in unordered_stats.items()})
        
        # Category-level statistics
        cat_stats = self._compute_category_statistics(ground_truth, generated, use_higher=False)
        stats.update({f'category_{k}': v for k, v in cat_stats.items() if k not in ['gt_category_dist', 'gen_category_dist']})
        
        if self.config.use_higher_category:
            higher_cat_stats = self._compute_category_statistics(ground_truth, generated, use_higher=True)
            stats.update({f'higher_category_{k}': v for k, v in higher_cat_stats.items() if k not in ['gt_category_dist', 'gen_category_dist']})
        
        # Optionally save sequences
        if self.config.save_generated_sequences:
            stats['ground_truth_tokens'] = ground_truth
            stats['generated_tokens'] = generated
        
        return stats
    
    def evaluate(self):
        """Run the complete statistical evaluation."""
        logger.info("="*80)
        logger.info("Starting Statistical Evaluation")
        logger.info("="*80)
        logger.info(f"Number of people: {self.config.num_people}")
        logger.info(f"Generations per person: {self.config.generations_per_person}")
        logger.info(f"Prefix progression: start={self.config.prefix_start}, jump={self.config.prefix_jump}")
        logger.info(f"Horizon mode: {self.config.horizon_mode}")
        
        horizons = self._get_horizons()
        logger.info(f"Horizons to evaluate: {horizons}")
        
        # Collect all statistics
        all_stats = []
        
        # Select random people
        total_people = len(self.dataset)
        selected_people = np.random.choice(
            total_people,
            size=min(self.config.num_people, total_people),
            replace=False
        )
        
        logger.info(f"Selected {len(selected_people)} people for evaluation")
        
        # Evaluate each person
        for person_idx in tqdm(selected_people, desc="Evaluating people"):
            # Load sequence
            item = self.dataset[int(person_idx)]
            x4 = item["input_ids"]  # (4, L)
            pm = item["padding_mask"]  # (L,)
            L_real = int(pm.sum().item())
            x4, pm = x4[:, :L_real], pm[:L_real]
            
            logger.info(f"\nPerson {person_idx}: sequence length = {L_real}")
            
            # For each horizon
            for horizon in horizons:
                # Get prefix lengths for this horizon
                prefix_lengths = self._get_prefix_lengths(L_real, horizon)
                logger.info(f"  Horizon {horizon}: {len(prefix_lengths)} prefix lengths")
                
                # For each prefix length
                for prefix_len in prefix_lengths:
                    # Generate multiple times
                    for gen_num in range(self.config.generations_per_person):
                        stats = self._evaluate_single_generation(
                            person_idx=person_idx,
                            generation_num=gen_num,
                            prefix_len=prefix_len,
                            horizon=horizon,
                            full_sequence=x4,
                            full_mask=pm
                        )
                        all_stats.append(stats)
        
        logger.info(f"\nCollected {len(all_stats)} evaluation records")
        
        # Save results
        self._save_results(all_stats)
        
        # Generate summary
        self._generate_summary(all_stats)
        
        logger.info("="*80)
        logger.info(f"Evaluation complete! Results saved to: {self.run_output_dir}")
        logger.info("="*80)
    
    def _save_results(self, all_stats: List[Dict]):
        """Save detailed results to CSV files."""
        logger.info("Saving results to CSV...")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_stats)
        
        # Remove complex columns for main CSV
        simple_df = df.drop(columns=[
            col for col in df.columns
            if 'tokens' in col or 'dist' in col
        ], errors='ignore')
        
        # Save main results
        main_csv = os.path.join(self.run_output_dir, f"{self.run_id}_detailed_stats.csv")
        simple_df.to_csv(main_csv, index=False)
        logger.info(f"Saved detailed statistics to: {main_csv}")
        
        # Save aggregated statistics by prefix length and horizon
        if len(simple_df) > 0:
            agg_by_prefix_horizon = simple_df.groupby(['prefix_len', 'horizon']).agg({
                col: ['mean', 'std', 'min', 'max']
                for col in simple_df.columns
                if col not in ['person_idx', 'generation_num', 'prefix_len', 'horizon']
                and simple_df[col].dtype in ['float64', 'int64']
            }).reset_index()
            
            agg_csv = os.path.join(self.run_output_dir, f"{self.run_id}_aggregated_by_prefix_horizon.csv")
            agg_by_prefix_horizon.to_csv(agg_csv, index=False)
            logger.info(f"Saved aggregated statistics to: {agg_csv}")
        
        # Save per-person aggregates
        if len(simple_df) > 0:
            per_person = simple_df.groupby('person_idx').agg({
                col: ['mean', 'std']
                for col in simple_df.columns
                if col not in ['person_idx', 'generation_num']
                and simple_df[col].dtype in ['float64', 'int64']
            }).reset_index()
            
            per_person_csv = os.path.join(self.run_output_dir, f"{self.run_id}_per_person_stats.csv")
            per_person.to_csv(per_person_csv, index=False)
            logger.info(f"Saved per-person statistics to: {per_person_csv}")
    
    def _generate_summary(self, all_stats: List[Dict]):
        """Generate summary report."""
        logger.info("Generating summary report...")
        
        df = pd.DataFrame(all_stats)
        
        summary = {
            'run_id': self.run_id,
            'config': asdict(self.config),
            'total_evaluations': len(all_stats),
            'unique_people': df['person_idx'].nunique() if len(df) > 0 else 0,
            'horizons_evaluated': sorted(df['horizon'].unique().tolist()) if len(df) > 0 else [],
        }
        
        # Add overall statistics
        if len(df) > 0:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            numeric_cols = [col for col in numeric_cols if col not in ['person_idx', 'generation_num', 'prefix_len', 'horizon']]
            
            for col in numeric_cols:
                summary[f'overall_{col}_mean'] = float(df[col].mean())
                summary[f'overall_{col}_std'] = float(df[col].std())
        
        # Save summary
        summary_path = os.path.join(self.run_output_dir, f"{self.run_id}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to: {summary_path}")
        
        # Print key metrics
        logger.info("\n" + "="*80)
        logger.info("KEY METRICS SUMMARY")
        logger.info("="*80)
        if len(df) > 0 and self.config.compute_ordered:
            logger.info(f"Overall Ordered Match Rate: {df['ordered_match_rate'].mean():.4f} ± {df['ordered_match_rate'].std():.4f}")
        if len(df) > 0 and self.config.compute_unordered:
            logger.info(f"Overall Jaccard Similarity: {df['unordered_jaccard_similarity'].mean():.4f} ± {df['unordered_jaccard_similarity'].std():.4f}")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Statistical Evaluation of Generative Models"
    )
    parser.add_argument(
        "--hparams",
        required=True,
        help="Path to evaluation hparams file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.hparams}")
    hparams = read_hparams(args.hparams)
    
    # Create config object
    config = EvalConfig(**hparams)
    
    # Create evaluator
    evaluator = StatisticalEvaluator(config)
    
    # Run evaluation
    evaluator.evaluate()


if __name__ == "__main__":
    main()
