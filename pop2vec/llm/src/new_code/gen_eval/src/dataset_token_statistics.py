#!/usr/bin/env python3
"""
Dataset Token Statistics - Analyze vocabulary and token usage across multiple datasets.

For each dataset (D0, D1, D2, D3, D4, etc.), this script:
1. Loads the vocabulary file (vocab.csv with columns: TOKEN, CATEGORY, ID)
2. Scans sequence files (encoded.h5) to compute:
   - n_people: How many people have this token in their sequence (unique)
   - n_observation: Total occurrences of this token across all sequences
3. (Optional) Extract 2D PCA coordinates from pretrained model embeddings
4. (Optional) Generate t-SNE visualizations for each model's embeddings

The script handles the folder structure:
    dataset_root/
        vocab.csv (or vocab_v0.csv)
        encoding=mlm/
            masking=random/
                encoded.h5
            masking=event/
                encoded.h5
        encoding=nomlm/
            masking=random/
                encoded.h5

Output:
    - Enhanced vocab CSV with columns like: mlm_random_n_people, mlm_random_n_observation, etc.
    - Metadata JSON with total people and observations per file
    - (Optional) Token embeddings PCA CSV with 2D coordinates for each model
    - (Optional) t-SNE plots for each model's embeddings
    - Token mismatch reports for cross-model comparisons

Usage:
    python dataset_token_statistics.py --config datasets.yaml --output_dir ./stats_output
    
    # Run only token statistics
    python dataset_token_statistics.py --config datasets.yaml --output_dir ./stats_output --only_token_stats
    
    # Run only model embeddings (requires --stats_file)
    python dataset_token_statistics.py --config datasets.yaml --output_dir ./stats_output \\
        --only_embeddings --stats_file ./stats_output/D0_vocab_stats.csv
    
    # Run only t-SNE visualization (requires embedding files to exist)
    python dataset_token_statistics.py --config datasets.yaml --output_dir ./stats_output --only_tsne
    
    # Or specify datasets directly
    python dataset_token_statistics.py \\
        --D0 /path/to/D0 \\
        --D1 /path/to/D1 \\
        --D3_parent_sibling /path/to/D3_parent_sibling \\
        --output_dir ./stats_output

t-SNE Perplexity Guide:
    - Range: 5-50 (must be < number of tokens)
    - Low (5-15): Focus on local structure, tight clusters
    - Medium (30): Default, good balance
    - High (50+): Focus on global structure
    - For vocab size ~5000-10000: use 30-50
    - For vocab size ~1000-5000: use 20-30
    - For vocab size <1000: use 10-20
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import h5py
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Optional imports for model embedding extraction
try:
    import torch
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_TORCH_SKLEARN = True
except ImportError:
    HAS_TORCH_SKLEARN = False

# Optional import for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logging with both file and console handlers."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"dataset_token_stats_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("dataset_token_stats")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = []  # Clear existing handlers
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - info level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_file}")
    return logger


# Global logger - will be initialized in main()
logger = logging.getLogger("dataset_token_stats")


# Default minimum people threshold for exportable tokens
DEFAULT_MIN_PEOPLE = 10


# Dataset configurations
@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    root_path: str
    vocab_file: str = "vocab.csv"  # or vocab_v0.csv


@dataclass
class ModelConfig:
    """Configuration for a pretrained model."""
    name: str  # Short name for the model
    path: str  # Path to model directory (containing hparams.yaml and model.ckpt)
    description: str = ""  # Optional description
    checkpoint_file: str = "model.ckpt"  # Checkpoint filename
    hparams_file: str = "hparams.yaml"  # Hyperparameters filename
    embedding_key: str = "transformer.embedding.token.parametrizations.weight.original"  # Key in state_dict
    dataset: str = ""  # Dataset name this model was trained on (optional, for vocab lookup)
    stats_file: str = ""  # Path to stats file for this model's vocab (optional)
    
    def get_hparams_path(self) -> Optional[str]:
        """Get full path to hparams file."""
        hparams_path = os.path.join(self.path, self.hparams_file)
        if os.path.exists(hparams_path):
            return hparams_path
        return None
    
    def get_checkpoint_path(self) -> str:
        """Get full path to checkpoint file."""
        return os.path.join(self.path, self.checkpoint_file)


@dataclass  
class SequenceFileInfo:
    """Information about a sequence file."""
    path: str
    encoding: str  # 'mlm' or 'nomlm'
    masking: str   # 'random' or 'event'
    prefix: str    # e.g., 'mlm_random', 'nomlm_random'


@dataclass
class TokenMismatchInfo:
    """Information about token mismatches between models/datasets."""
    token_name: str
    models_with_token: List[str] = field(default_factory=list)
    models_without_token: List[str] = field(default_factory=list)
    id_by_model: Dict[str, int] = field(default_factory=dict)
    category_by_model: Dict[str, str] = field(default_factory=dict)
    has_id_mismatch: bool = False
    has_category_mismatch: bool = False


class TokenMismatchTracker:
    """Track and report token mismatches across models."""
    
    def __init__(self):
        self.token_info: Dict[str, TokenMismatchInfo] = {}
        self.model_tokens: Dict[str, Set[str]] = {}  # model_name -> set of token names
        self.model_vocab_size: Dict[str, int] = {}
        
    def add_model_vocab(self, model_name: str, vocab_df: pd.DataFrame):
        """Register a model's vocabulary."""
        tokens = set(vocab_df['TOKEN'].tolist())
        self.model_tokens[model_name] = tokens
        self.model_vocab_size[model_name] = len(vocab_df)
        
        for _, row in vocab_df.iterrows():
            token_name = row['TOKEN']
            token_id = row.get('ID', -1)
            category = row.get('CATEGORY', 'unknown')
            
            if token_name not in self.token_info:
                self.token_info[token_name] = TokenMismatchInfo(token_name=token_name)
            
            info = self.token_info[token_name]
            info.models_with_token.append(model_name)
            info.id_by_model[model_name] = token_id
            info.category_by_model[model_name] = str(category)
    
    def analyze_mismatches(self) -> Dict:
        """Analyze and return mismatch statistics."""
        all_models = list(self.model_tokens.keys())
        all_tokens = set(self.token_info.keys())
        
        # Tokens present in all models
        common_tokens = all_tokens.copy()
        for model_tokens in self.model_tokens.values():
            common_tokens &= model_tokens
        
        # Analyze each token
        id_mismatches = []
        category_mismatches = []
        model_specific_tokens = defaultdict(list)  # model -> tokens only in that model
        
        for token_name, info in self.token_info.items():
            # Check which models don't have this token
            info.models_without_token = [m for m in all_models if m not in info.id_by_model]
            
            # Check for ID mismatches
            unique_ids = set(info.id_by_model.values())
            if len(unique_ids) > 1:
                info.has_id_mismatch = True
                id_mismatches.append(info)
            
            # Check for category mismatches
            unique_categories = set(info.category_by_model.values())
            if len(unique_categories) > 1:
                info.has_category_mismatch = True
                category_mismatches.append(info)
            
            # Track model-specific tokens
            if len(info.models_with_token) == 1:
                model_specific_tokens[info.models_with_token[0]].append(token_name)
        
        return {
            'total_unique_tokens': len(all_tokens),
            'common_tokens': len(common_tokens),
            'id_mismatches': id_mismatches,
            'category_mismatches': category_mismatches,
            'model_specific_tokens': dict(model_specific_tokens),
            'model_vocab_sizes': self.model_vocab_size.copy(),
        }
    
    def generate_report(self, output_dir: str) -> str:
        """Generate detailed mismatch report and save to files."""
        analysis = self.analyze_mismatches()
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("TOKEN MISMATCH REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary
        report_lines.append("SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total unique tokens across all models: {analysis['total_unique_tokens']}")
        report_lines.append(f"Common tokens (in all models): {analysis['common_tokens']}")
        report_lines.append(f"Tokens with ID mismatches: {len(analysis['id_mismatches'])}")
        report_lines.append(f"Tokens with category mismatches: {len(analysis['category_mismatches'])}")
        report_lines.append("")
        
        # Model vocab sizes
        report_lines.append("MODEL VOCABULARY SIZES")
        report_lines.append("-" * 40)
        for model, size in analysis['model_vocab_sizes'].items():
            report_lines.append(f"  {model}: {size:,} tokens")
        report_lines.append("")
        
        # Model-specific tokens
        report_lines.append("MODEL-SPECIFIC TOKENS (only in one model)")
        report_lines.append("-" * 40)
        for model, tokens in analysis['model_specific_tokens'].items():
            report_lines.append(f"  {model}: {len(tokens)} unique tokens")
            if len(tokens) <= 20:
                for t in tokens:
                    report_lines.append(f"    - {t}")
            else:
                for t in tokens[:10]:
                    report_lines.append(f"    - {t}")
                report_lines.append(f"    ... and {len(tokens) - 10} more")
        report_lines.append("")
        
        # ID mismatches (detailed)
        if analysis['id_mismatches']:
            report_lines.append("TOKEN ID MISMATCHES")
            report_lines.append("-" * 40)
            report_lines.append("(Same token name has different IDs across models)")
            for info in analysis['id_mismatches'][:50]:  # Limit to first 50
                report_lines.append(f"  Token: {info.token_name}")
                for model, tid in info.id_by_model.items():
                    report_lines.append(f"    {model}: ID={tid}")
            if len(analysis['id_mismatches']) > 50:
                report_lines.append(f"  ... and {len(analysis['id_mismatches']) - 50} more")
            report_lines.append("")
        
        # Category mismatches
        if analysis['category_mismatches']:
            report_lines.append("TOKEN CATEGORY MISMATCHES")
            report_lines.append("-" * 40)
            for info in analysis['category_mismatches'][:50]:
                report_lines.append(f"  Token: {info.token_name}")
                for model, cat in info.category_by_model.items():
                    report_lines.append(f"    {model}: CATEGORY={cat}")
            if len(analysis['category_mismatches']) > 50:
                report_lines.append(f"  ... and {len(analysis['category_mismatches']) - 50} more")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = os.path.join(output_dir, "token_mismatch_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Save detailed CSV of ID mismatches
        if analysis['id_mismatches']:
            mismatch_data = []
            for info in analysis['id_mismatches']:
                row = {'TOKEN': info.token_name}
                for model, tid in info.id_by_model.items():
                    row[f'{model}_ID'] = tid
                for model, cat in info.category_by_model.items():
                    row[f'{model}_CATEGORY'] = cat
                mismatch_data.append(row)
            
            mismatch_df = pd.DataFrame(mismatch_data)
            mismatch_csv_path = os.path.join(output_dir, "token_id_mismatches.csv")
            mismatch_df.to_csv(mismatch_csv_path, index=False)
            logger.info(f"Saved ID mismatch details: {mismatch_csv_path}")
        
        # Save token presence matrix
        presence_data = []
        for token_name, info in self.token_info.items():
            row = {'TOKEN': token_name}
            for model in self.model_tokens.keys():
                row[f'{model}_present'] = model in info.id_by_model
                if model in info.id_by_model:
                    row[f'{model}_ID'] = info.id_by_model[model]
            presence_data.append(row)
        
        presence_df = pd.DataFrame(presence_data)
        presence_csv_path = os.path.join(output_dir, "token_presence_matrix.csv")
        presence_df.to_csv(presence_csv_path, index=False)
        logger.info(f"Saved token presence matrix: {presence_csv_path}")
        
        # Log summary
        logger.info("=" * 60)
        logger.info("TOKEN MISMATCH SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total unique tokens: {analysis['total_unique_tokens']}")
        logger.info(f"Common tokens: {analysis['common_tokens']}")
        logger.info(f"ID mismatches: {len(analysis['id_mismatches'])}")
        logger.info(f"Category mismatches: {len(analysis['category_mismatches'])}")
        logger.info(f"Report saved: {report_path}")
        
        return report_path


# Global mismatch tracker
mismatch_tracker = TokenMismatchTracker()


# =============================================================================
# Embedding Extraction Functions
# =============================================================================

def extract_token_embeddings(model_path: str, hparams_path: Optional[str] = None) -> Tuple[np.ndarray, Optional[str]]:
    """
    Extract token embeddings from a pretrained model checkpoint.
    
    Tries multiple common paths for the embedding weights:
    - transformer.embedding.token.parametrizations.weight.original
    - transformer.embedding.token.weight
    - embedding.token.weight
    - model.embedding.weight
    
    Args:
        model_path: Path to the model checkpoint (.ckpt file)
        hparams_path: Optional path to hparams.yaml for vocab info
    
    Returns:
        Tuple of (embeddings array, vocab_path from hparams or None)
    """
    try:
        import torch
    except ImportError:
        logger.error("PyTorch is required for embedding extraction. Install with: pip install torch")
        return np.array([]), None
    
    logger.info(f"Loading model checkpoint: {model_path}")
    
    try:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return np.array([]), None
    
    # Get state dict
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        logger.error(f"Unexpected checkpoint format")
        return np.array([]), None
    
    # Try different paths for embedding weights
    embedding_keys = [
        'transformer.embedding.token.parametrizations.weight.original',
        'transformer.embedding.token.weight',
        'embedding.token.weight',
        'model.embedding.weight',
        'embeddings.word_embeddings.weight',
        'encoder.embed_tokens.weight',
    ]
    
    embeddings = None
    for key in embedding_keys:
        if key in state_dict:
            embeddings = state_dict[key]
            logger.info(f"Found embeddings at: {key}")
            break
    
    if embeddings is None:
        # Try to find any key containing 'embedding' and 'weight'
        for key in state_dict.keys():
            if 'embedding' in key.lower() and 'weight' in key.lower():
                embeddings = state_dict[key]
                logger.info(f"Found embeddings at: {key}")
                break
    
    if embeddings is None:
        logger.error(f"Could not find embedding weights in checkpoint")
        logger.info(f"Available keys: {list(state_dict.keys())[:20]}...")
        return np.array([]), None
    
    # Convert to numpy
    if hasattr(embeddings, 'numpy'):
        embeddings = embeddings.numpy()
    elif hasattr(embeddings, 'cpu'):
        embeddings = embeddings.cpu().numpy()
    
    logger.info(f"Embedding shape: {embeddings.shape}")
    
    # Try to get vocab path from hparams
    vocab_path = None
    if hparams_path and os.path.exists(hparams_path):
        try:
            with open(hparams_path, 'r') as f:
                hparams = yaml.safe_load(f)
            vocab_path = hparams.get('vocab_path')
            logger.info(f"Vocab path from hparams: {vocab_path}")
        except Exception as e:
            logger.warning(f"Failed to load hparams: {e}")
    
    return embeddings, vocab_path


def compute_pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute 2D PCA of embeddings.
    
    Args:
        embeddings: (vocab_size, embedding_dim) array
    
    Returns:
        (vocab_size, 2) array of 2D coordinates
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        logger.error("scikit-learn is required for PCA. Install with: pip install scikit-learn")
        return np.zeros((len(embeddings), 2))
    
    logger.info(f"Computing PCA for {len(embeddings)} tokens...")
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embeddings)
    
    logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    return coords_2d


def compute_tsne_2d(embeddings: np.ndarray, perplexity: int = 30, random_state: int = 42) -> np.ndarray:
    """
    Compute 2D t-SNE of embeddings.
    
    Args:
        embeddings: (vocab_size, embedding_dim) array
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
    
    Returns:
        (vocab_size, 2) array of 2D coordinates
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        logger.error("scikit-learn is required for t-SNE. Install with: pip install scikit-learn")
        return np.zeros((len(embeddings), 2))
    
    logger.info(f"Computing t-SNE for {len(embeddings)} tokens (this may take a while)...")
    
    # Adjust perplexity if too high for the number of samples
    actual_perplexity = min(perplexity, len(embeddings) - 1)
    if actual_perplexity < perplexity:
        logger.warning(f"Reduced perplexity from {perplexity} to {actual_perplexity} due to small sample size")
    
    tsne = TSNE(n_components=2, perplexity=actual_perplexity, random_state=random_state, n_jobs=-1)
    coords_2d = tsne.fit_transform(embeddings)
    
    return coords_2d


def load_stats_for_vocab(
    vocab_df: pd.DataFrame,
    stats_file: Optional[str],
    output_dir: str,
    dataset_name: str = ""
) -> pd.DataFrame:
    """
    Load and merge token statistics into vocab DataFrame.
    
    Matches tokens by TOKEN name (not ID) to handle different token ID assignments
    across datasets.
    
    Args:
        vocab_df: Vocabulary DataFrame with TOKEN, CATEGORY, ID columns
        stats_file: Path to stats CSV file, or None
        output_dir: Output directory to search for stats files
        dataset_name: Dataset name to find stats file automatically
    
    Returns:
        DataFrame with stats columns merged (n_people, n_observation, exportable)
    """
    result_df = vocab_df.copy()
    
    # Try to find stats file
    if stats_file and os.path.exists(stats_file):
        logger.info(f"Loading stats from: {stats_file}")
        stats_df = pd.read_csv(stats_file)
    elif dataset_name:
        # Try to find stats file in output directory
        possible_stats = os.path.join(output_dir, f'{dataset_name}_vocab_stats.csv')
        if os.path.exists(possible_stats):
            logger.info(f"Loading stats from: {possible_stats}")
            stats_df = pd.read_csv(possible_stats)
        else:
            logger.warning(f"No stats file found for dataset {dataset_name}")
            return result_df
    else:
        logger.warning("No stats file provided and no dataset name for lookup")
        return result_df
    
    # Find n_people and n_observation columns in stats
    n_people_cols = [c for c in stats_df.columns if c.endswith('_n_people')]
    n_obs_cols = [c for c in stats_df.columns if c.endswith('_n_observation')]
    
    if not n_people_cols:
        logger.warning("No n_people columns found in stats file")
        return result_df
    
    # Merge by TOKEN name (not ID) to handle different ID assignments
    # Take the first n_people column as the primary one
    merge_cols = ['TOKEN'] + n_people_cols + n_obs_cols
    
    # Only keep columns that exist
    merge_cols = [c for c in merge_cols if c in stats_df.columns]
    
    stats_subset = stats_df[merge_cols].drop_duplicates(subset=['TOKEN'])
    
    # Merge on TOKEN
    result_df = result_df.merge(stats_subset, on='TOKEN', how='left')
    
    # Fill NaN with 0
    for col in n_people_cols + n_obs_cols:
        if col in result_df.columns:
            result_df[col] = result_df[col].fillna(0).astype(int)
    
    # Add exportable flag based on first n_people column
    if n_people_cols:
        primary_n_people = n_people_cols[0]
        result_df['exportable'] = result_df[primary_n_people] >= DEFAULT_MIN_PEOPLE
        logger.info(f"Exportable tokens: {result_df['exportable'].sum()} / {len(result_df)}")
    
    return result_df


def process_model_embeddings(
    model: ModelConfig,
    vocab_df: pd.DataFrame,
    stats_df: Optional[pd.DataFrame],
    output_dir: str,
    min_people: int = DEFAULT_MIN_PEOPLE
) -> Optional[pd.DataFrame]:
    """
    Process a single model and extract 2D PCA embeddings with stats.
    
    Args:
        model: Model configuration
        vocab_df: Vocabulary DataFrame with TOKEN, CATEGORY, ID columns
        stats_df: Optional DataFrame with token statistics (n_people, n_observation)
        output_dir: Output directory
        min_people: Minimum n_people for exportable flag
    
    Returns:
        DataFrame with columns: TOKEN, CATEGORY, ID, x, y, model_name, model_description,
                               n_people, n_observation, exportable
        Or None if failed
    """
    logger.info(f"Processing model: {model.name}")
    
    model_path = os.path.join(model.path, model.checkpoint_file)
    
    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found: {model_path}")
        return None
    
    # Get hparams path
    hparams_path = model.get_hparams_path()
    
    # Extract embeddings
    embeddings, vocab_path_from_hparams = extract_token_embeddings(model_path, hparams_path)
    
    if len(embeddings) == 0:
        return None
    
    # Verify vocab size matches
    working_vocab_df = vocab_df.copy()
    if len(embeddings) != len(working_vocab_df):
        logger.warning(f"Embedding size ({len(embeddings)}) doesn't match vocab size ({len(working_vocab_df)})")
        # Try to use vocab from hparams if available
        if vocab_path_from_hparams and os.path.exists(vocab_path_from_hparams):
            logger.info(f"Loading vocab from hparams: {vocab_path_from_hparams}")
            working_vocab_df = pd.read_csv(vocab_path_from_hparams)
            if len(embeddings) != len(working_vocab_df):
                logger.error(f"Still mismatch after loading hparams vocab")
                return None
    
    # Compute 2D PCA
    coords_2d = compute_pca_2d(embeddings)
    
    # Create result DataFrame
    result_df = working_vocab_df.copy()
    result_df['x'] = coords_2d[:, 0]
    result_df['y'] = coords_2d[:, 1]
    result_df['model_name'] = model.name
    result_df['model_description'] = model.description
    result_df['model_dataset'] = model.dataset
    
    # Merge stats if available
    if stats_df is not None:
        # Find n_people and n_observation columns
        n_people_cols = [c for c in stats_df.columns if c.endswith('_n_people')]
        n_obs_cols = [c for c in stats_df.columns if c.endswith('_n_observation')]
        
        if n_people_cols:
            # Merge by TOKEN name
            merge_cols = ['TOKEN'] + n_people_cols + n_obs_cols
            merge_cols = [c for c in merge_cols if c in stats_df.columns]
            stats_subset = stats_df[merge_cols].drop_duplicates(subset=['TOKEN'])
            
            result_df = result_df.merge(stats_subset, on='TOKEN', how='left')
            
            # Fill NaN with 0
            for col in n_people_cols + n_obs_cols:
                if col in result_df.columns:
                    result_df[col] = result_df[col].fillna(0).astype(int)
            
            # Add exportable flag
            primary_n_people = n_people_cols[0]
            result_df['exportable'] = result_df[primary_n_people] >= min_people
            logger.info(f"Exportable tokens: {result_df['exportable'].sum()} / {len(result_df)}")
    
    return result_df


def extract_all_model_embeddings(
    models: List[ModelConfig],
    datasets: Dict[str, 'DatasetConfig'],
    output_dir: str,
    stats_file: Optional[str] = None,
    min_people: int = DEFAULT_MIN_PEOPLE
) -> List[str]:
    """
    Extract embeddings from all models and create individual CSV files.
    
    Also tracks token mismatches across models and generates:
    - Individual model embedding files
    - Combined files split by token compatibility
    - Mismatch reports
    
    Args:
        models: List of model configurations
        datasets: Dict of dataset_name -> DatasetConfig
        output_dir: Output directory
        stats_file: Optional global stats file to use for all models
        min_people: Minimum n_people for exportable flag
    
    Returns:
        List of paths to individual model embedding CSVs
    """
    global mismatch_tracker
    mismatch_tracker = TokenMismatchTracker()  # Reset tracker
    
    logger.info("="*70)
    logger.info("EXTRACTING MODEL EMBEDDINGS")
    logger.info("="*70)
    logger.info(f"Processing {len(models)} model(s)")
    
    output_files = []
    all_embeddings_dfs = []  # List of (model_name, DataFrame)
    
    start_time = time.time()
    
    for model in tqdm(models, desc="Extracting embeddings", unit="model"):
        model_start = time.time()
        logger.info(f"\n--- Processing model: {model.name} ---")
        logger.debug(f"Model path: {model.path}")
        logger.debug(f"Dataset: {model.dataset}")
        
        vocab_df = None
        stats_df = None
        
        # Find the vocab for this model's dataset
        if model.dataset and model.dataset in datasets:
            dataset = datasets[model.dataset]
            vocab_path = find_vocab_file(dataset.root_path)
            if vocab_path:
                logger.info(f"Loading vocab from: {vocab_path}")
                vocab_df = pd.read_csv(vocab_path)
                logger.info(f"Vocab size: {len(vocab_df)}")
                
                # Try to load stats for this dataset
                dataset_stats_path = os.path.join(output_dir, f'{model.dataset}_vocab_stats.csv')
                if model.stats_file and os.path.exists(model.stats_file):
                    logger.info(f"Loading stats from model config: {model.stats_file}")
                    stats_df = pd.read_csv(model.stats_file)
                elif os.path.exists(dataset_stats_path):
                    logger.info(f"Loading stats from dataset: {dataset_stats_path}")
                    stats_df = pd.read_csv(dataset_stats_path)
                elif stats_file and os.path.exists(stats_file):
                    logger.info(f"Loading stats from global file: {stats_file}")
                    stats_df = pd.read_csv(stats_file)
                else:
                    logger.warning(f"No stats file found for model {model.name}")
            else:
                logger.warning(f"No vocab found for dataset {model.dataset}, skipping model {model.name}")
                continue
        else:
            # Try to get vocab from model's hparams
            hparams_path = model.get_hparams_path()
            if hparams_path:
                try:
                    logger.info(f"Loading hparams from: {hparams_path}")
                    with open(hparams_path, 'r') as f:
                        hparams = yaml.safe_load(f)
                    vocab_path = hparams.get('vocab_path')
                    if vocab_path and os.path.exists(vocab_path):
                        logger.info(f"Loading vocab from hparams: {vocab_path}")
                        vocab_df = pd.read_csv(vocab_path)
                        logger.info(f"Vocab size: {len(vocab_df)}")
                        
                        # Use provided stats file
                        if model.stats_file and os.path.exists(model.stats_file):
                            stats_df = pd.read_csv(model.stats_file)
                        elif stats_file and os.path.exists(stats_file):
                            stats_df = pd.read_csv(stats_file)
                    else:
                        logger.warning(f"No vocab path in hparams for model {model.name}")
                        continue
                except Exception as e:
                    logger.warning(f"Failed to load hparams for model {model.name}: {e}")
                    continue
            else:
                logger.warning(f"No dataset or hparams for model {model.name}, skipping")
                continue
        
        # Track vocab for mismatch analysis
        mismatch_tracker.add_model_vocab(model.name, vocab_df)
        
        # Process model
        result_df = process_model_embeddings(model, vocab_df, stats_df, output_dir, min_people)
        
        if result_df is not None:
            # Save individual model embeddings
            model_output_path = os.path.join(output_dir, f'{model.name}_embeddings_pca.csv')
            result_df.to_csv(model_output_path, index=False)
            logger.info(f"Saved: {model_output_path}")
            output_files.append(model_output_path)
            all_embeddings_dfs.append((model.name, result_df))
            
            model_elapsed = time.time() - model_start
            logger.info(f"Model {model.name} completed in {model_elapsed:.1f}s")
    
    # Generate combined files with split by token compatibility
    if len(all_embeddings_dfs) > 1:
        logger.info("\n" + "="*70)
        logger.info("GENERATING COMBINED FILES")
        logger.info("="*70)
        
        # Get common tokens across all models
        all_tokens_sets = [set(df['TOKEN'].tolist()) for _, df in all_embeddings_dfs]
        common_tokens = set.intersection(*all_tokens_sets) if all_tokens_sets else set()
        all_tokens = set.union(*all_tokens_sets) if all_tokens_sets else set()
        
        logger.info(f"Total unique tokens across all models: {len(all_tokens)}")
        logger.info(f"Common tokens (in all models): {len(common_tokens)}")
        
        # Part 1: Common tokens only (safe to combine)
        if common_tokens:
            combined_common = []
            for model_name, df in all_embeddings_dfs:
                common_df = df[df['TOKEN'].isin(common_tokens)].copy()
                combined_common.append(common_df)
            
            combined_common_df = pd.concat(combined_common, ignore_index=True)
            common_path = os.path.join(output_dir, 'combined_embeddings_common_tokens.csv')
            combined_common_df.to_csv(common_path, index=False)
            logger.info(f"Saved common tokens combined file: {common_path}")
            logger.info(f"  {len(common_tokens)} tokens x {len(all_embeddings_dfs)} models = {len(combined_common_df)} rows")
        
        # Part 2: All tokens (may have mismatches)
        combined_all = pd.concat([df for _, df in all_embeddings_dfs], ignore_index=True)
        all_path = os.path.join(output_dir, 'combined_embeddings_all_tokens.csv')
        combined_all.to_csv(all_path, index=False)
        logger.info(f"Saved all tokens combined file: {all_path}")
        logger.info(f"  {len(combined_all)} total rows")
        
        # Part 3: Model-specific tokens (tokens unique to each model)
        for model_name, df in all_embeddings_dfs:
            other_tokens = set()
            for other_name, other_df in all_embeddings_dfs:
                if other_name != model_name:
                    other_tokens.update(other_df['TOKEN'].tolist())
            
            unique_tokens = set(df['TOKEN'].tolist()) - other_tokens
            if unique_tokens:
                unique_df = df[df['TOKEN'].isin(unique_tokens)].copy()
                unique_path = os.path.join(output_dir, f'{model_name}_unique_tokens.csv')
                unique_df.to_csv(unique_path, index=False)
                logger.info(f"Saved {model_name} unique tokens: {unique_path} ({len(unique_tokens)} tokens)")
    
    # Generate mismatch report
    if len(all_embeddings_dfs) > 1:
        mismatch_tracker.generate_report(output_dir)
    
    total_elapsed = time.time() - start_time
    logger.info(f"\nTotal embedding extraction time: {total_elapsed:.1f}s")
    
    if output_files:
        logger.info(f"Generated {len(output_files)} embedding file(s)")
    else:
        logger.warning("No embeddings extracted")
    
    return output_files


# =============================================================================
# t-SNE Visualization Functions
# =============================================================================

def plot_tsne_embeddings(
    embeddings_df: pd.DataFrame,
    embeddings_array: np.ndarray,
    model_name: str,
    output_dir: str,
    category_column: str = 'CATEGORY',
    min_people: int = DEFAULT_MIN_PEOPLE,
    perplexity: int = 30
) -> Optional[str]:
    """
    Create t-SNE visualization for model embeddings.
    
    Args:
        embeddings_df: DataFrame with TOKEN, CATEGORY, ID, and optionally n_people columns
        embeddings_array: Raw embedding vectors (vocab_size, embedding_dim)
        model_name: Name of the model
        output_dir: Output directory for plots
        category_column: Column name for coloring points
        min_people: Minimum n_people for highlighted points
        perplexity: t-SNE perplexity
    
    Returns:
        Path to saved plot, or None if failed
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib is required for plotting. Install with: pip install matplotlib")
        return None
    
    if not HAS_TORCH_SKLEARN:
        logger.error("scikit-learn is required for t-SNE. Install with: pip install scikit-learn")
        return None
    
    logger.info(f"Generating t-SNE plot for model: {model_name}")
    
    # Compute t-SNE
    tsne_coords = compute_tsne_2d(embeddings_array, perplexity=perplexity)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Check if we have category information
    if category_column in embeddings_df.columns:
        categories = embeddings_df[category_column].unique()
        cmap = plt.cm.get_cmap('tab20', len(categories))
        
        for i, cat in enumerate(categories):
            mask = embeddings_df[category_column] == cat
            ax.scatter(
                tsne_coords[mask, 0],
                tsne_coords[mask, 1],
                c=[cmap(i)],
                label=cat if len(str(cat)) < 20 else str(cat)[:17] + '...',
                alpha=0.6,
                s=10
            )
        
        # Add legend with smaller font if many categories
        if len(categories) <= 15:
            ax.legend(loc='upper right', fontsize=8)
    else:
        # No category, use single color
        ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1], alpha=0.6, s=10)
    
    # Highlight exportable tokens if we have n_people info
    n_people_cols = [c for c in embeddings_df.columns if c.endswith('_n_people')]
    if n_people_cols:
        primary_n_people = n_people_cols[0]
        exportable_mask = embeddings_df[primary_n_people] >= min_people
        
        ax.scatter(
            tsne_coords[exportable_mask, 0],
            tsne_coords[exportable_mask, 1],
            facecolors='none',
            edgecolors='red',
            s=50,
            linewidths=1,
            label=f'Exportable (n≥{min_people})'
        )
    
    ax.set_title(f't-SNE Visualization: {model_name}', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{model_name}_tsne.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved t-SNE plot: {plot_path}")
    
    # Also save t-SNE coordinates to CSV
    tsne_df = embeddings_df.copy()
    tsne_df['tsne_x'] = tsne_coords[:, 0]
    tsne_df['tsne_y'] = tsne_coords[:, 1]
    tsne_csv_path = os.path.join(output_dir, f'{model_name}_embeddings_tsne.csv')
    tsne_df.to_csv(tsne_csv_path, index=False)
    logger.info(f"Saved t-SNE coordinates: {tsne_csv_path}")
    
    return plot_path


def generate_tsne_plots(
    models: List[ModelConfig],
    output_dir: str,
    perplexity: int = 30
) -> List[str]:
    """
    Generate t-SNE plots for all models using existing embedding files.
    
    Args:
        models: List of model configurations
        output_dir: Output directory (should contain *_embeddings_pca.csv files)
        perplexity: t-SNE perplexity
    
    Returns:
        List of paths to generated plot files
    """
    logger.info("="*70)
    logger.info("GENERATING t-SNE VISUALIZATIONS")
    logger.info("="*70)
    
    plot_files = []
    
    for model in tqdm(models, desc="Generating t-SNE plots"):
        # Check for embedding PCA file
        pca_file = os.path.join(output_dir, f'{model.name}_embeddings_pca.csv')
        
        if not os.path.exists(pca_file):
            logger.warning(f"PCA file not found for {model.name}, skipping t-SNE")
            continue
        
        # Load PCA embeddings DataFrame
        embeddings_df = pd.read_csv(pca_file)
        
        # Load raw embeddings from checkpoint
        model_path = os.path.join(model.path, model.checkpoint_file)
        if not os.path.exists(model_path):
            logger.warning(f"Checkpoint not found for {model.name}, skipping t-SNE")
            continue
        
        hparams_path = model.get_hparams_path()
        embeddings_array, _ = extract_token_embeddings(model_path, hparams_path)
        
        if len(embeddings_array) == 0:
            logger.warning(f"Failed to extract embeddings for {model.name}")
            continue
        
        if len(embeddings_array) != len(embeddings_df):
            logger.warning(f"Embedding size mismatch for {model.name}: {len(embeddings_array)} vs {len(embeddings_df)}")
            continue
        
        # Generate plot
        plot_path = plot_tsne_embeddings(
            embeddings_df,
            embeddings_array,
            model.name,
            output_dir,
            perplexity=perplexity
        )
        
        if plot_path:
            plot_files.append(plot_path)
    
    if plot_files:
        logger.info(f"Generated {len(plot_files)} t-SNE plot(s)")
    else:
        logger.warning("No t-SNE plots generated")
    
    return plot_files


def find_vocab_file(root_path: str) -> Optional[str]:
    """Find vocabulary file in the dataset root."""
    possible_names = ['vocab.csv', 'vocab_v0.csv', 'vocabulary.csv']
    
    for name in possible_names:
        path = os.path.join(root_path, name)
        if os.path.exists(path):
            return path
    
    return None


def find_sequence_files(root_path: str) -> List[SequenceFileInfo]:
    """Find all encoded.h5 files in the dataset structure."""
    files = []
    
    # Check for encoding folders
    for encoding in ['mlm', 'nomlm']:
        encoding_dir = os.path.join(root_path, f'encoding={encoding}')
        
        if not os.path.exists(encoding_dir):
            continue
        
        # Check for masking folders
        for masking in ['random', 'event']:
            masking_dir = os.path.join(encoding_dir, f'masking={masking}')
            h5_path = os.path.join(masking_dir, 'encoded.h5')
            
            if os.path.exists(h5_path):
                prefix = f'{encoding}_{masking}'
                files.append(SequenceFileInfo(
                    path=h5_path,
                    encoding=encoding,
                    masking=masking,
                    prefix=prefix
                ))
        
        # Also check directly in encoding folder (for datasets without masking subfolder)
        direct_h5 = os.path.join(encoding_dir, 'encoded.h5')
        if os.path.exists(direct_h5):
            # Use 'random' as default masking for nomlm
            prefix = f'{encoding}_random'
            files.append(SequenceFileInfo(
                path=direct_h5,
                encoding=encoding,
                masking='random',
                prefix=prefix
            ))
    
    # Also check for h5 file directly in root
    root_h5 = os.path.join(root_path, 'encoded.h5')
    if os.path.exists(root_h5):
        files.append(SequenceFileInfo(
            path=root_h5,
            encoding='unknown',
            masking='unknown',
            prefix='default'
        ))
    
    return files


def process_chunk_for_token_stats(
    h5_path: str,
    start_idx: int,
    end_idx: int,
    pad_id: int = 0
) -> Tuple[Counter, Counter]:
    """
    Process a chunk of sequences and count token occurrences.
    
    Returns:
        Tuple of (n_people_counter, n_observation_counter)
        - n_people_counter: token_id -> number of sequences containing this token
        - n_observation_counter: token_id -> total occurrences
    """
    n_people_counter = Counter()
    n_observation_counter = Counter()
    
    try:
        with h5py.File(h5_path, 'r') as f:
            input_ids = f['input_ids']
            
            # Handle different data shapes
            # Could be (N, 4, seq_len) or (N, seq_len)
            if len(input_ids.shape) == 3:
                # Shape: (N, 4, seq_len) - tokens are at index 0
                tokens = input_ids[start_idx:end_idx, 0, :]
            else:
                # Shape: (N, seq_len)
                tokens = input_ids[start_idx:end_idx, :]
            
            for seq_idx in range(tokens.shape[0]):
                seq_tokens = tokens[seq_idx, :]
                
                # Exclude PAD tokens
                valid_tokens = seq_tokens[seq_tokens != pad_id]
                
                # Count unique tokens for n_people (each person counted once per token type)
                unique_tokens = set(valid_tokens.tolist())
                for token_id in unique_tokens:
                    n_people_counter[token_id] += 1
                
                # Count all occurrences for n_observation
                for token_id in valid_tokens:
                    n_observation_counter[int(token_id)] += 1
    
    except Exception as e:
        logger.error(f"Error processing chunk [{start_idx}:{end_idx}] in {h5_path}: {e}")
    
    return n_people_counter, n_observation_counter


def compute_token_statistics(
    h5_path: str,
    n_workers: int = 8,
    chunk_size: int = 50000,
    pad_id: int = 0
) -> Tuple[Counter, Counter, int, int]:
    """
    Compute token statistics for a sequence file.
    
    Returns:
        Tuple of (n_people_counter, n_observation_counter, total_people, total_observations)
    """
    logger.info(f"Processing: {h5_path}")
    
    # Get dataset info
    with h5py.File(h5_path, 'r') as f:
        if 'input_ids' not in f:
            logger.warning(f"No 'input_ids' key in {h5_path}")
            return Counter(), Counter(), 0, 0
        
        input_ids = f['input_ids']
        n_sequences = input_ids.shape[0]
        
        if len(input_ids.shape) == 3:
            seq_len = input_ids.shape[2]
        else:
            seq_len = input_ids.shape[1]
        
        logger.info(f"  Shape: {input_ids.shape}, sequences: {n_sequences:,}, seq_len: {seq_len}")
    
    # Create chunks
    chunks = []
    for start in range(0, n_sequences, chunk_size):
        end = min(start + chunk_size, n_sequences)
        chunks.append((start, end))
    
    # Process chunks in parallel
    all_n_people = Counter()
    all_n_observation = Counter()
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_chunk_for_token_stats, h5_path, start, end, pad_id): (start, end)
            for start, end in chunks
        }
        
        with tqdm(total=len(chunks), desc=f"  Chunks", leave=False) as pbar:
            for future in as_completed(futures):
                n_people_chunk, n_obs_chunk = future.result()
                all_n_people.update(n_people_chunk)
                all_n_observation.update(n_obs_chunk)
                pbar.update(1)
    
    elapsed = time.time() - start_time
    total_people = n_sequences
    total_observations = sum(all_n_observation.values())
    
    logger.info(f"  Complete in {elapsed:.1f}s")
    logger.info(f"  Total people: {total_people:,}")
    logger.info(f"  Total observations: {total_observations:,}")
    logger.info(f"  Unique tokens: {len(all_n_observation):,}")
    
    return all_n_people, all_n_observation, total_people, total_observations


def process_dataset(
    dataset: DatasetConfig,
    output_dir: str,
    n_workers: int = 8,
    chunk_size: int = 50000,
    pad_id: int = 0
) -> Optional[str]:
    """
    Process a single dataset and create enhanced vocabulary file.
    
    Returns path to output vocab file, or None if failed.
    """
    logger.info("="*70)
    logger.info(f"Processing Dataset: {dataset.name}")
    logger.info(f"Root: {dataset.root_path}")
    logger.info("="*70)
    
    if not os.path.exists(dataset.root_path):
        logger.error(f"Dataset root not found: {dataset.root_path}")
        return None
    
    # Find vocabulary file
    vocab_path = find_vocab_file(dataset.root_path)
    if vocab_path is None:
        logger.error(f"No vocabulary file found in {dataset.root_path}")
        return None
    
    logger.info(f"Vocabulary file: {vocab_path}")
    
    # Load vocabulary
    vocab_df = pd.read_csv(vocab_path)
    logger.info(f"Vocabulary size: {len(vocab_df)}")
    
    # Find sequence files
    seq_files = find_sequence_files(dataset.root_path)
    
    if not seq_files:
        logger.warning(f"No sequence files found in {dataset.root_path}")
        # Just copy the vocab file as-is
        output_path = os.path.join(output_dir, f'{dataset.name}_vocab_stats.csv')
        vocab_df.to_csv(output_path, index=False)
        return output_path
    
    logger.info(f"Found {len(seq_files)} sequence file(s):")
    for sf in seq_files:
        logger.info(f"  - {sf.prefix}: {sf.path}")
    
    # Process each sequence file
    metadata = {
        'dataset_name': dataset.name,
        'root_path': dataset.root_path,
        'vocab_path': vocab_path,
        'vocab_size': len(vocab_df),
        'sequence_files': {}
    }
    
    for seq_file in tqdm(seq_files, desc=f"Processing {dataset.name} files"):
        n_people_counter, n_obs_counter, total_people, total_obs = compute_token_statistics(
            seq_file.path,
            n_workers=n_workers,
            chunk_size=chunk_size,
            pad_id=pad_id
        )
        
        # Add columns to vocab dataframe
        col_n_people = f'{seq_file.prefix}_n_people'
        col_n_obs = f'{seq_file.prefix}_n_observation'
        
        vocab_df[col_n_people] = vocab_df['ID'].map(lambda x: n_people_counter.get(x, 0))
        vocab_df[col_n_obs] = vocab_df['ID'].map(lambda x: n_obs_counter.get(x, 0))
        
        # Store metadata
        metadata['sequence_files'][seq_file.prefix] = {
            'path': seq_file.path,
            'encoding': seq_file.encoding,
            'masking': seq_file.masking,
            'total_people': total_people,
            'total_observations': total_obs,
            'unique_tokens_used': len(n_obs_counter),
        }
    
    # Save enhanced vocab
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{dataset.name}_vocab_stats.csv')
    vocab_df.to_csv(output_path, index=False)
    logger.info(f"Saved enhanced vocab: {output_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f'{dataset.name}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {metadata_path}")
    
    return output_path


def main():
    # Pre-parse output_dir for early logging setup
    import sys
    output_dir_for_logging = "./output"
    for i, arg in enumerate(sys.argv):
        if arg == "--output_dir" and i + 1 < len(sys.argv):
            output_dir_for_logging = sys.argv[i + 1]
            break
    
    # Setup logging early
    global logger
    logger = setup_logging(output_dir_for_logging)
    logger.info("Starting Dataset Token Statistics")
    
    parser = argparse.ArgumentParser(
        description="Compute token statistics, extract model embeddings, and generate t-SNE visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using config file (runs all layers: token stats, embeddings, t-SNE)
    python dataset_token_statistics.py --config datasets.yaml --output_dir ./stats_output
    
    # Run only token statistics
    python dataset_token_statistics.py --config datasets.yaml --output_dir ./stats_output --only_token_stats
    
    # Run only model embeddings extraction (requires --stats_file for export filtering)
    python dataset_token_statistics.py --config datasets.yaml --output_dir ./stats_output \\
        --only_embeddings --stats_file ./stats_output/D0_vocab_stats.csv
    
    # Run only t-SNE visualization (requires embedding files to exist)
    python dataset_token_statistics.py --config datasets.yaml --output_dir ./stats_output --only_tsne
    
    # Specifying datasets directly
    python dataset_token_statistics.py \\
        --D0 /path/to/D0 \\
        --D1 /path/to/D1 \\
        --D3_parent_sibling /path/to/D3_parent_sibling \\
        --output_dir ./stats_output

Config file format (YAML):
    datasets:
      D0: /path/to/D0
      D1: /path/to/D1
      D2: /path/to/D2
      D3_parent_sibling: /path/to/D3_parent_sibling
      D3_full_pop: /path/to/D3_full_pop
      D4: /path/to/D4
      D4_bd: /path/to/D4_bd
    
    models:
      - name: model_v1
        path: /path/to/model_v1
        description: "First version of the model"
        checkpoint_file: model.ckpt  # optional, defaults to model.ckpt
        hparams_file: hparams.yaml   # optional, defaults to hparams.yaml
        dataset: D0                  # optional, which dataset's vocab to use
        stats_file: /path/to/stats.csv  # optional, for export filtering
      - name: model_v2
        path: /path/to/model_v2
        description: "Second version"
        """
    )
    
    # Config file option
    parser.add_argument("--config", help="Path to YAML config file with dataset paths")
    
    # Direct dataset path options
    parser.add_argument("--D0", help="Path to D0 dataset root")
    parser.add_argument("--D1", help="Path to D1 dataset root")
    parser.add_argument("--D2", help="Path to D2 dataset root")
    parser.add_argument("--D3_parent_sibling", help="Path to D3_parent_sibling dataset root")
    parser.add_argument("--D3_full_pop", help="Path to D3_full_pop dataset root")
    parser.add_argument("--D4", help="Path to D4 dataset root")
    parser.add_argument("--D4_bd", help="Path to D4_bd dataset root")
    
    # Control which layers to run
    parser.add_argument("--only_token_stats", action="store_true",
                        help="Run only token statistics (Layer 1)")
    parser.add_argument("--only_embeddings", action="store_true",
                        help="Run only model embeddings extraction (Layer 2)")
    parser.add_argument("--only_tsne", action="store_true",
                        help="Run only t-SNE visualization (Layer 3, requires embeddings to exist)")
    
    # Stats file for embedding export filtering
    parser.add_argument("--stats_file", 
                        help="Path to stats CSV file for embedding export filtering (used with --only_embeddings)")
    
    # General options
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--n_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--chunk_size", type=int, default=50000, help="Chunk size for processing")
    parser.add_argument("--pad_id", type=int, default=0, help="PAD token ID")
    parser.add_argument("--min_people", type=int, default=DEFAULT_MIN_PEOPLE,
                        help=f"Minimum n_people for exportable tokens (default: {DEFAULT_MIN_PEOPLE})")
    parser.add_argument("--tsne_perplexity", type=int, default=30,
                        help="t-SNE perplexity parameter (default: 30)")
    
    args = parser.parse_args()
    
    # Validate mutually exclusive options
    exclusive_flags = [args.only_token_stats, args.only_embeddings, args.only_tsne]
    if sum(exclusive_flags) > 1:
        logger.error("Cannot use multiple --only_* flags together")
        return
    
    run_token_stats = not args.only_embeddings and not args.only_tsne
    run_embeddings = not args.only_token_stats and not args.only_tsne
    run_tsne = not args.only_token_stats and not args.only_embeddings
    
    # If only_* is specified, only run that layer
    if args.only_token_stats:
        run_embeddings = False
        run_tsne = False
    elif args.only_embeddings:
        run_token_stats = False
        run_tsne = False
    elif args.only_tsne:
        run_token_stats = False
        run_embeddings = False
    
    # Build list of datasets and models
    datasets = []
    datasets_dict = {}
    models = []
    
    # From config file
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Parse datasets
        for name, path in config.get('datasets', {}).items():
            if path and path.strip():
                ds = DatasetConfig(name=name, root_path=path)
                datasets.append(ds)
                datasets_dict[name] = ds
        
        # Parse models
        for model_cfg in config.get('models', []):
            if isinstance(model_cfg, dict):
                model = ModelConfig(
                    name=model_cfg.get('name', 'unnamed'),
                    path=model_cfg.get('path', ''),
                    description=model_cfg.get('description', ''),
                    checkpoint_file=model_cfg.get('checkpoint_file', 'model.ckpt'),
                    hparams_file=model_cfg.get('hparams_file', 'hparams.yaml'),
                    embedding_key=model_cfg.get('embedding_key', 
                        'transformer.embedding.token.parametrizations.weight.original'),
                    dataset=model_cfg.get('dataset', ''),
                    stats_file=model_cfg.get('stats_file', '')
                )
                models.append(model)
    
    # From command line arguments
    dataset_args = ['D0', 'D1', 'D2', 'D3_parent_sibling', 'D3_full_pop', 'D4', 'D4_bd']
    for name in dataset_args:
        path = getattr(args, name, None)
        if path and path.strip():
            # Check if already added from config
            if not any(d.name == name for d in datasets):
                ds = DatasetConfig(name=name, root_path=path)
                datasets.append(ds)
                datasets_dict[name] = ds
    
    # Validation
    if run_token_stats and not datasets:
        logger.warning("No datasets specified for token statistics.")
        if not args.only_token_stats:
            logger.info("Skipping token statistics layer.")
            run_token_stats = False
        else:
            logger.error("No datasets specified. Use --config or --D0, --D1, etc.")
            return
    
    if (run_embeddings or run_tsne) and not models:
        logger.warning("No models specified for embedding extraction/t-SNE.")
        if args.only_embeddings or args.only_tsne:
            logger.error("No models specified. Add 'models' section to config file.")
            return
        logger.info("Skipping embeddings and t-SNE layers.")
        run_embeddings = False
        run_tsne = False
    
    if (run_embeddings or run_tsne) and not HAS_TORCH_SKLEARN:
        logger.error("PyTorch and scikit-learn are required for embedding extraction.")
        logger.error("Install with: pip install torch scikit-learn")
        if args.only_embeddings or args.only_tsne:
            return
        logger.info("Skipping embeddings and t-SNE layers.")
        run_embeddings = False
        run_tsne = False
    
    if run_tsne and not HAS_MATPLOTLIB:
        logger.error("matplotlib is required for t-SNE visualization.")
        logger.error("Install with: pip install matplotlib")
        if args.only_tsne:
            return
        logger.info("Skipping t-SNE layer.")
        run_tsne = False
    
    if not run_token_stats and not run_embeddings and not run_tsne:
        logger.error("Nothing to do. Specify datasets and/or models in config.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log what we'll do
    logger.info("="*70)
    logger.info("DATASET TOKEN STATISTICS AND EMBEDDINGS")
    logger.info("="*70)
    
    if run_token_stats:
        logger.info(f"Layer 1 - Token Statistics: {len(datasets)} dataset(s)")
        for ds in datasets:
            logger.info(f"  - {ds.name}: {ds.root_path}")
    
    if run_embeddings:
        logger.info(f"Layer 2 - Model Embeddings: {len(models)} model(s)")
        for m in models:
            logger.info(f"  - {m.name}: {m.path}")
    
    if run_tsne:
        logger.info(f"Layer 3 - t-SNE Visualization: {len(models)} model(s)")
    
    # Process token statistics (Layer 1)
    results = []
    if run_token_stats:
        logger.info("\n" + "="*70)
        logger.info("LAYER 1: TOKEN STATISTICS")
        logger.info("="*70)
        
        for dataset in tqdm(datasets, desc="Datasets"):
            output_path = process_dataset(
                dataset,
                args.output_dir,
                n_workers=args.n_workers,
                chunk_size=args.chunk_size,
                pad_id=args.pad_id
            )
            if output_path:
                results.append((dataset.name, output_path))
    
    # Extract model embeddings (Layer 2)
    embedding_files = []
    if run_embeddings:
        logger.info("\n" + "="*70)
        logger.info("LAYER 2: MODEL EMBEDDINGS")
        logger.info("="*70)
        
        embedding_files = extract_all_model_embeddings(
            models, 
            datasets_dict, 
            args.output_dir,
            stats_file=args.stats_file,
            min_people=args.min_people
        )
    
    # Generate t-SNE visualizations (Layer 3)
    tsne_files = []
    if run_tsne:
        logger.info("\n" + "="*70)
        logger.info("LAYER 3: t-SNE VISUALIZATION")
        logger.info("="*70)
        
        tsne_files = generate_tsne_plots(
            models,
            args.output_dir,
            perplexity=args.tsne_perplexity
        )
    
    # Create summary
    logger.info("\n" + "="*70)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*70)
    
    if results:
        logger.info("Token Statistics:")
        for name, path in results:
            logger.info(f"  {name}: {path}")
    
    if embedding_files:
        logger.info(f"Embedding Files ({len(embedding_files)}):")
        for path in embedding_files:
            logger.info(f"  {path}")
    
    if tsne_files:
        logger.info(f"t-SNE Files ({len(tsne_files)}):")
        for path in tsne_files:
            logger.info(f"  {path}")
    
    # Create combined summary
    summary_path = os.path.join(args.output_dir, 'all_datasets_summary.json')
    summary = {
        'datasets_processed': len(results),
        'datasets': {name: path for name, path in results},
        'models_processed': len(models) if run_embeddings else 0,
        'embedding_files': embedding_files if embedding_files else [],
        'tsne_files': tsne_files if tsne_files else [],
        'output_dir': args.output_dir,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved: {summary_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
