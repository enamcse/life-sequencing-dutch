#!/usr/bin/env python3
"""
Statistical Evaluation of Generative Model

This script performs comprehensive statistical evaluation of generative models by:
1. Testing on multiple real people's sequences
2. Generating multiple times per person (for statistical significance)
3. Varying prefix lengths progressively
4. Measuring both ordered and unordered matches
5. Computing statistics by token, category, and higher category
6. Supporting single or multiple generation horizons
7. Outputting sequences in multiple formats (token_id, token, category, higher_category)

Usage:
    python generative_statistical_evaluation.py --hparams path/to/eval_hparams.txt

Output:
    - Detailed CSV with all generations and counts
    - Aggregated statistics by prefix/horizon
    - Per-person aggregated stats
    - Optional: 4 formats of generated sequences (token_id, token, category, higher_category)
    - Summary report with count-based metrics
"""

import argparse
import csv
import json
import logging
import numpy as np
import os
import pandas as pd
import time
import torch
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from tqdm import tqdm
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
    num_people: int = 10
    generations_per_person: int = 100
    
    # Prefix progression
    prefix_start: int = 1
    prefix_jump: int = 100
    
    # Generation horizons
    horizon_mode: str = "single"
    horizon_single: int = 20
    horizon_gap: int = 20
    horizon_max: int = 200
    
    # Matching modes
    compute_ordered: bool = True
    compute_unordered: bool = True
    
    # Categorization
    use_category: bool = True
    use_higher_category: bool = True
    
    # Sampling
    top_k: int = 20
    temperature: float = 1.0
    
    # Special tokens
    pad_token: str = "[PAD]"
    cls_token: str = "[CLS]"
    death_token: str = "[DEATH]"
    
    # Output control
    save_generated_sequences: bool = True
    save_token_id_format: bool = True
    save_token_format: bool = True
    save_category_format: bool = True
    save_higher_category_format: bool = True
    registry_mode: str = "subfolder"


class StatisticalEvaluator:
    """Main evaluator class for statistical analysis."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Load vocabulary
        logger.info(f"Loading vocabulary from: {config.vocab_path}")
        self.vocab_df = load_vocab_df(config.vocab_path)
        
        # Check columns
        if "CATEGORY" not in self.vocab_df.columns:
            raise ValueError("CATEGORY column missing from vocabulary")
        
        if config.use_higher_category and "HIGHER_CATEGORY" not in self.vocab_df.columns:
            logger.warning("HIGHER_CATEGORY not found, disabling it")
            self.config.use_higher_category = False
        
        # Create mappings
        self.id_to_token = dict(zip(self.vocab_df['ID'], self.vocab_df['TOKEN']))
        self.id_to_category = dict(zip(self.vocab_df['ID'], self.vocab_df['CATEGORY']))
        if self.config.use_higher_category:
            self.id_to_higher_category = dict(zip(self.vocab_df['ID'], self.vocab_df['HIGHER_CATEGORY']))
        
        # Load special tokens
        specials = load_special_ids(
            config.vocab_path,
            pad_token=config.pad_token,
            cls_token=config.cls_token,
            death_token=config.death_token,
        )
        self.pad_id = specials["pad_id"]
        self.cls_id = specials["cls_id"]
        self.death_id = specials["death_id"]
        
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
        
        # Setup output
        self.run_id = f"run_{int(time.time())}"
        self._setup_output_structure()
    
    def _setup_output_structure(self):
        """Setup output directory structure."""
        if self.config.registry_mode == "subfolder":
            self.run_output_dir = os.path.join(self.config.output_dir, self.run_id)
            os.makedirs(self.run_output_dir, exist_ok=True)
            
            with open(os.path.join(self.run_output_dir, "config.json"), 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            
            logger.info(f"Created run directory: {self.run_output_dir}")
        else:
            self.run_output_dir = self.config.output_dir
            registry_path = os.path.join(self.config.output_dir, "registry.csv")
            
            with open(registry_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not os.path.exists(registry_path) or os.path.getsize(registry_path) == 0:
                    writer.writerow(['run_id', 'timestamp', 'config_json'])
                writer.writerow([self.run_id, time.time(), json.dumps(asdict(self.config))])
    
    def _get_horizons(self) -> List[int]:
        """Get list of horizons to evaluate."""
        if self.config.horizon_mode == "single":
            return [self.config.horizon_single]
        else:
            horizons = []
            h = self.config.horizon_gap
            while h <= self.config.horizon_max:
                horizons.append(h)
                h += self.config.horizon_gap
            return horizons
    
    def _get_prefix_lengths(self, max_seq_len: int, horizon: int) -> List[int]:
        """Get list of prefix lengths."""
        prefix_lengths = [self.config.prefix_start]
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
        x = prefix_4stream.unsqueeze(0).to(self.device)
        pm = pad_mask.unsqueeze(0).to(self.device)
        out_tokens = []
        
        for _ in range(horizon):
            logits = self.model({"input_ids": x, "padding_mask": pm})
            last_logits = logits[:, -1, :] / max(1e-8, float(self.config.temperature))
            
            if self.config.top_k > 0:
                vals, idxs = torch.topk(last_logits, k=self.config.top_k, dim=-1)
                probs = softmax(vals, dim=-1)
                next_token = idxs.gather(-1, torch.multinomial(probs, 1)).squeeze(-1)
            else:
                next_token = torch.argmax(last_logits, dim=-1)
            
            tid = int(next_token.item())
            out_tokens.append(tid)
            
            if self.death_id is not None and tid == self.death_id:
                break
            
            # Append new token
            last_age = x[0, 1, -1].item()
            last_day = x[0, 2, -1].item()
            new_step = torch.tensor([[tid],[last_age],[last_day],[1]], dtype=torch.long, device=self.device)
            x = torch.cat([x, new_step.unsqueeze(0)], dim=2)
            pm = torch.cat([pm, torch.ones(1,1, dtype=pm.dtype, device=self.device)], dim=1)
        
        return out_tokens
    
    def _compute_matches(
        self,
        ground_truth: List[int],
        generated: List[int]
    ) -> Dict:
        """Compute all matching statistics."""
        stats = {}
        
        # ORDERED MATCHES (position-based)
        if self.config.compute_ordered:
            min_len = min(len(ground_truth), len(generated))
            
            # Token-level ordered
            cnt_ordered_token = sum(
                1 for i in range(min_len) if ground_truth[i] == generated[i]
            ) if min_len > 0 else 0
            stats['cnt_ordered_token_match'] = cnt_ordered_token
            stats['total_positions'] = min_len
            
            # Category-level ordered
            if self.config.use_category:
                gt_cats = [self.id_to_category.get(tid, "UNK") for tid in ground_truth]
                gen_cats = [self.id_to_category.get(tid, "UNK") for tid in generated]
                cnt_ordered_cat = sum(
                    1 for i in range(min_len) if gt_cats[i] == gen_cats[i]
                ) if min_len > 0 else 0
                stats['cnt_ordered_category_match'] = cnt_ordered_cat
            
            # Higher category-level ordered
            if self.config.use_higher_category:
                gt_hcats = [self.id_to_higher_category.get(tid, "UNK") for tid in ground_truth]
                gen_hcats = [self.id_to_higher_category.get(tid, "UNK") for tid in generated]
                cnt_ordered_hcat = sum(
                    1 for i in range(min_len) if gt_hcats[i] == gen_hcats[i]
                ) if min_len > 0 else 0
                stats['cnt_ordered_higher_category_match'] = cnt_ordered_hcat
        
        # UNORDERED MATCHES (set-based)
        # CRITICAL FIX: Count ALL occurrences, not just unique
        if self.config.compute_unordered:
            # Token-level unordered - count min occurrences
            gt_counter = Counter(ground_truth)
            gen_counter = Counter(generated)
            cnt_unordered_token = sum(
                min(gt_counter[tid], gen_counter[tid])
                for tid in set(ground_truth) | set(generated)
            )
            stats['cnt_unordered_token_match'] = cnt_unordered_token
            stats['total_gt_tokens'] = len(ground_truth)
            stats['total_gen_tokens'] = len(generated)
            
            # Category-level unordered
            if self.config.use_category:
                gt_cats = [self.id_to_category.get(tid, "UNK") for tid in ground_truth]
                gen_cats = [self.id_to_category.get(tid, "UNK") for tid in generated]
                gt_cat_counter = Counter(gt_cats)
                gen_cat_counter = Counter(gen_cats)
                cnt_unordered_cat = sum(
                    min(gt_cat_counter[c], gen_cat_counter[c])
                    for c in set(gt_cats) | set(gen_cats)
                )
                stats['cnt_unordered_category_match'] = cnt_unordered_cat
            
            # Higher category-level unordered
            if self.config.use_higher_category:
                gt_hcats = [self.id_to_higher_category.get(tid, "UNK") for tid in ground_truth]
                gen_hcats = [self.id_to_higher_category.get(tid, "UNK") for tid in generated]
                gt_hcat_counter = Counter(gt_hcats)
                gen_hcat_counter = Counter(gen_hcats)
                cnt_unordered_hcat = sum(
                    min(gt_hcat_counter[c], gen_hcat_counter[c])
                    for c in set(gt_hcats) | set(gen_hcats)
                )
                stats['cnt_unordered_higher_category_match'] = cnt_unordered_hcat
        
        return stats
    
    def _ids_to_format(self, ids: List[int], format_type: str) -> str:
        """Convert token IDs to specified format."""
        if format_type == "token_id":
            return ",".join(str(tid) for tid in ids)
        elif format_type == "token":
            return ",".join(self.id_to_token.get(tid, f"<UNK:{tid}>") for tid in ids)
        elif format_type == "category":
            return ",".join(self.id_to_category.get(tid, "UNK") for tid in ids)
        elif format_type == "higher_category":
            return ",".join(self.id_to_higher_category.get(tid, "UNK") for tid in ids)
        else:
            raise ValueError(f"Unknown format: {format_type}")
    
    def _evaluate_single_generation(
        self,
        generation_num: int,
        prefix_len: int,
        horizon: int,
        full_sequence: torch.Tensor,
        full_mask: torch.Tensor,
        rinpersoon_id: int
    ) -> Dict:
        """Evaluate a single generation."""
        # Extract prefix
        prefix_4stream = full_sequence[:, :prefix_len]
        prefix_mask = full_mask[:prefix_len]
        
        # Extract ground truth continuation
        ground_truth_end = min(prefix_len + horizon, full_sequence.size(1))
        ground_truth = full_sequence[0, prefix_len:ground_truth_end].tolist()
        
        # Generate
        generated = self._generate_tokens(prefix_4stream, prefix_mask, horizon)
        
        # Compute matches
        stats = self._compute_matches(ground_truth, generated)
        
        # Add metadata
        stats.update({
            'rinpersoon_id': rinpersoon_id,
            'generation_num': generation_num,
            'prefix_len': prefix_len,
            'horizon': horizon,
            'cnt_orig_prefix_token': prefix_len,
            'cnt_orig_continuation_token': len(ground_truth),
            'cnt_generated_token': len(generated),
        })
        
        # Store sequences for later format conversion
        if self.config.save_generated_sequences:
            prefix_tokens = full_sequence[0, :prefix_len].tolist()
            stats['_prefix_ids'] = prefix_tokens
            stats['_gt_ids'] = ground_truth
            stats['_gen_ids'] = generated
        
        return stats
    
    def evaluate(self):
        """Run complete statistical evaluation."""
        logger.info("="*80)
        logger.info("Starting Statistical Evaluation")
        logger.info("="*80)
        
        horizons = self._get_horizons()
        logger.info(f"Horizons: {horizons}")
        
        # Select people
        total_people = len(self.dataset)
        selected_people = np.random.choice(
            total_people,
            size=min(self.config.num_people, total_people),
            replace=False
        )
        logger.info(f"Selected {len(selected_people)} people")
        
        # Collect statistics
        all_stats = []
        
        for person_idx in tqdm(selected_people, desc="Evaluating people"):
            item = self.dataset[person_idx]
            x4 = item["input_ids"]
            pm = item["padding_mask"]
            L_real = int(pm.sum().item())
            x4, pm = x4[:, :L_real], pm[:L_real]
            
            rinpersoon_id = item.get("rinpersoon_id", person_idx)
            
            for horizon in horizons:
                prefix_lengths = self._get_prefix_lengths(L_real, horizon)
                
                for prefix_len in prefix_lengths:
                    for gen_num in range(self.config.generations_per_person):
                        stats = self._evaluate_single_generation(
                            gen_num, prefix_len, horizon,
                            x4, pm, rinpersoon_id
                        )
                        all_stats.append(stats)
        
        logger.info(f"Collected {len(all_stats)} evaluation records")
        
        # Save results
        self._save_results(all_stats)
        self._generate_summary(all_stats)
        
        logger.info("="*80)
        logger.info(f"Results saved to: {self.run_output_dir}")
        logger.info("="*80)
    
    def _save_results(self, all_stats: List[Dict]):
        """Save detailed results."""
        logger.info("Saving results...")
        
        df = pd.DataFrame(all_stats)
        
        # 1. DETAILED STATS - Only counts, no derived metrics
        detail_cols = [
            'rinpersoon_id', 'generation_num', 'prefix_len', 'horizon',
            'cnt_orig_prefix_token', 'cnt_orig_continuation_token', 'cnt_generated_token'
        ]
        
        if self.config.compute_ordered:
            detail_cols.extend(['cnt_ordered_token_match', 'total_positions'])
            if self.config.use_category:
                detail_cols.append('cnt_ordered_category_match')
            if self.config.use_higher_category:
                detail_cols.append('cnt_ordered_higher_category_match')
        
        if self.config.compute_unordered:
            detail_cols.extend(['cnt_unordered_token_match', 'total_gt_tokens', 'total_gen_tokens'])
            if self.config.use_category:
                detail_cols.append('cnt_unordered_category_match')
            if self.config.use_higher_category:
                detail_cols.append('cnt_unordered_higher_category_match')
        
        detail_df = df[detail_cols]
        detail_csv = os.path.join(self.run_output_dir, f"{self.run_id}_detailed_stats.csv")
        detail_df.to_csv(detail_csv, index=False)
        logger.info(f"Saved detailed stats to: {detail_csv}")
        
        # 2. SEQUENCE FILES (if enabled) - 4 separate formats
        if self.config.save_generated_sequences and '_prefix_ids' in df.columns:
            seq_base_cols = [
                'rinpersoon_id', 'generation_num', 'prefix_len', 'horizon',
                'cnt_orig_prefix_token', 'cnt_orig_continuation_token', 'cnt_generated_token'
            ]
            
            if self.config.compute_ordered:
                seq_base_cols.append('cnt_ordered_token_match')
                if self.config.use_category:
                    seq_base_cols.append('cnt_ordered_category_match')
                if self.config.use_higher_category:
                    seq_base_cols.append('cnt_ordered_higher_category_match')
            
            if self.config.compute_unordered:
                seq_base_cols.append('cnt_unordered_token_match')
                if self.config.use_category:
                    seq_base_cols.append('cnt_unordered_category_match')
                if self.config.use_higher_category:
                    seq_base_cols.append('cnt_unordered_higher_category_match')
            
            # Format 1: token_id
            if self.config.save_token_id_format:
                seq_df = df[seq_base_cols].copy()
                seq_df['original_prefix'] = df['_prefix_ids'].apply(
                    lambda x: ','.join(str(i) for i in x)
                )
                seq_df['original_continuation'] = df['_gt_ids'].apply(
                    lambda x: ','.join(str(i) for i in x)
                )
                seq_df['generated_tokens'] = df['_gen_ids'].apply(
                    lambda x: ','.join(str(i) for i in x)
                )
                seq_csv = os.path.join(self.run_output_dir, f"{self.run_id}_sequences_token_id.csv")
                seq_df.to_csv(seq_csv, index=False)
                logger.info(f"Saved token_id sequences to: {seq_csv}")
            
            # Format 2: token
            if self.config.save_token_format:
                seq_df = df[seq_base_cols].copy()
                seq_df['original_prefix'] = df['_prefix_ids'].apply(
                    lambda ids: ','.join(self.id_to_token.get(i, f"<UNK:{i}>") for i in ids)
                )
                seq_df['original_continuation'] = df['_gt_ids'].apply(
                    lambda ids: ','.join(self.id_to_token.get(i, f"<UNK:{i}>") for i in ids)
                )
                seq_df['generated_tokens'] = df['_gen_ids'].apply(
                    lambda ids: ','.join(self.id_to_token.get(i, f"<UNK:{i}>") for i in ids)
                )
                seq_csv = os.path.join(self.run_output_dir, f"{self.run_id}_sequences_token.csv")
                seq_df.to_csv(seq_csv, index=False)
                logger.info(f"Saved token sequences to: {seq_csv}")
            
            # Format 3: category
            if self.config.save_category_format and self.config.use_category:
                seq_df = df[seq_base_cols].copy()
                seq_df['original_prefix'] = df['_prefix_ids'].apply(
                    lambda ids: ','.join(self.id_to_category.get(i, "UNK") for i in ids)
                )
                seq_df['original_continuation'] = df['_gt_ids'].apply(
                    lambda ids: ','.join(self.id_to_category.get(i, "UNK") for i in ids)
                )
                seq_df['generated_tokens'] = df['_gen_ids'].apply(
                    lambda ids: ','.join(self.id_to_category.get(i, "UNK") for i in ids)
                )
                seq_csv = os.path.join(self.run_output_dir, f"{self.run_id}_sequences_category.csv")
                seq_df.to_csv(seq_csv, index=False)
                logger.info(f"Saved category sequences to: {seq_csv}")
            
            # Format 4: higher_category
            if self.config.save_higher_category_format and self.config.use_higher_category:
                seq_df = df[seq_base_cols].copy()
                seq_df['original_prefix'] = df['_prefix_ids'].apply(
                    lambda ids: ','.join(self.id_to_higher_category.get(i, "UNK") for i in ids)
                )
                seq_df['original_continuation'] = df['_gt_ids'].apply(
                    lambda ids: ','.join(self.id_to_higher_category.get(i, "UNK") for i in ids)
                )
                seq_df['generated_tokens'] = df['_gen_ids'].apply(
                    lambda ids: ','.join(self.id_to_higher_category.get(i, "UNK") for i in ids)
                )
                seq_csv = os.path.join(self.run_output_dir, f"{self.run_id}_sequences_higher_category.csv")
                seq_df.to_csv(seq_csv, index=False)
                logger.info(f"Saved higher_category sequences to: {seq_csv}")
        
        # 3. AGGREGATED BY PREFIX_LEN AND HORIZON
        if len(df) > 0 and self.config.compute_ordered:
            group_cols = ['prefix_len', 'horizon']
            agg_dict = {
                'rinpersoon_id': 'count',  # num_evaluations
            }
            
            if self.config.compute_ordered:
                agg_dict['cnt_ordered_token_match'] = ['sum', 'mean', 'std', 'min', 'max']
                agg_dict['total_positions'] = 'sum'
            
            if self.config.compute_unordered:
                agg_dict['cnt_unordered_token_match'] = ['sum', 'mean', 'std', 'min', 'max']
                agg_dict['total_gt_tokens'] = 'sum'
            
            agg_df = df.groupby(group_cols).agg(agg_dict)
            agg_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                             for col in agg_df.columns.values]
            agg_df = agg_df.reset_index()
            agg_df.rename(columns={'rinpersoon_id': 'num_evaluations'}, inplace=True)
            
            agg_csv = os.path.join(self.run_output_dir, f"{self.run_id}_aggregated_by_prefix_horizon.csv")
            agg_df.to_csv(agg_csv, index=False)
            logger.info(f"Saved aggregated stats to: {agg_csv}")
        
        # 4. AGGREGATED BY PERSON
        if len(df) > 0:
            person_agg_dict = {'generation_num': 'count'}
            
            if self.config.compute_ordered:
                person_agg_dict['cnt_ordered_token_match'] = ['sum', 'mean', 'std']
                person_agg_dict['total_positions'] = 'sum'
            
            if self.config.compute_unordered:
                person_agg_dict['cnt_unordered_token_match'] = ['sum', 'mean', 'std']
                person_agg_dict['total_gt_tokens'] = 'sum'
            
            person_df = df.groupby('rinpersoon_id').agg(person_agg_dict)
            person_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                                for col in person_df.columns.values]
            person_df = person_df.reset_index()
            person_df.rename(columns={'generation_num': 'num_generations'}, inplace=True)
            
            person_csv = os.path.join(self.run_output_dir, f"{self.run_id}_aggregated_by_person.csv")
            person_df.to_csv(person_csv, index=False)
            logger.info(f"Saved per-person stats to: {person_csv}")
    
    def _generate_summary(self, all_stats: List[Dict]):
        """Generate summary report."""
        logger.info("Generating summary...")
        
        df = pd.DataFrame(all_stats)
        
        summary = {
            'run_id': self.run_id,
            'total_evaluations': len(all_stats),
            'unique_people': df['rinpersoon_id'].nunique() if len(df) > 0 else 0,
        }
        
        if len(df) > 0 and self.config.compute_ordered:
            total_pos = df['total_positions'].sum()
            total_ordered_matches = df['cnt_ordered_token_match'].sum()
            summary['ordered_token_matches_total'] = int(total_ordered_matches)
            summary['ordered_token_positions_total'] = int(total_pos)
            summary['ordered_token_match_rate'] = float(total_ordered_matches / total_pos if total_pos > 0 else 0)
        
        if len(df) > 0 and self.config.compute_unordered:
            total_gt = df['total_gt_tokens'].sum()
            total_unordered_matches = df['cnt_unordered_token_match'].sum()
            summary['unordered_token_matches_total'] = int(total_unordered_matches)
            summary['unordered_gt_tokens_total'] = int(total_gt)
            summary['unordered_token_match_rate'] = float(total_unordered_matches / total_gt if total_gt > 0 else 0)
        
        # Save summary
        summary_path = os.path.join(self.run_output_dir, f"{self.run_id}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to: {summary_path}")
        
        # Print key metrics
        logger.info("\n" + "="*80)
        logger.info("KEY METRICS SUMMARY")
        logger.info("="*80)
        if 'ordered_token_matches_total' in summary:
            logger.info(f"Ordered Token Matches: {summary['ordered_token_matches_total']} / {summary['ordered_token_positions_total']} ({summary['ordered_token_match_rate']:.4f})")
        if 'unordered_token_matches_total' in summary:
            logger.info(f"Unordered Token Matches: {summary['unordered_token_matches_total']} / {summary['unordered_gt_tokens_total']} ({summary['unordered_token_match_rate']:.4f})")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="Statistical Evaluation of Generative Models")
    parser.add_argument("--hparams", required=True, help="Path to evaluation hparams file")
    args = parser.parse_args()
    
    logger.info(f"Loading configuration from: {args.hparams}")
    hparams = read_hparams(args.hparams)
    
    config = EvalConfig(**hparams)
    evaluator = StatisticalEvaluator(config)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
