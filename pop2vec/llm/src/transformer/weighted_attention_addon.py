"""
Weighted Attention Extensions for Clinical Sequences
Provides token-level importance weighting based on:
- Temporal recency (recent events matter more)
- Token type/category (critical events like death, major diagnoses)
- Manual importance scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalImportanceWeighting(nn.Module):
    """
    Apply exponential decay to older tokens - recent events matter more.
    
    Args:
        decay_rate: How quickly importance decays (0.0 = no decay, 1.0 = strong decay)
        learnable: If True, decay_rate is a learnable parameter
    
    Example:
        Given sequence with ages [10, 15, 20, 25, 30]
        With decay_rate=0.1, more recent tokens (age 30) get higher weight
    """
    def __init__(self, decay_rate: float = 0.1, learnable: bool = False):
        super().__init__()
        if learnable:
            self.decay_rate = nn.Parameter(torch.tensor(decay_rate))
        else:
            self.register_buffer('decay_rate', torch.tensor(decay_rate))
    
    def forward(self, ages: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ages: (batch, seq_len) - age values from input_ids[:, 2, :]
        
        Returns:
            weights: (batch, seq_len) - importance weights
        """
        # Normalize ages to [0, 1] range within each sequence
        max_age = ages.max(dim=1, keepdim=True)[0]
        normalized_ages = ages / (max_age + 1e-8)
        
        # Exponential decay: more recent (higher age) = higher weight
        weights = torch.exp(self.decay_rate * normalized_ages)
        
        # Normalize to sum to 1 per sequence
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return weights


class CategoryBasedWeighting(nn.Module):
    """
    Weight tokens based on their category/type.
    
    Args:
        vocab_size: Total vocabulary size
        category_weights: Dict mapping token_id -> importance weight
        default_weight: Weight for tokens not in category_weights
    
    Example:
        category_weights = {
            DEATH_TOKEN_ID: 10.0,   # Death is very important
            HOSPITAL_ADMISSION: 5.0, # Hospital admissions important
            ROUTINE_CHECKUP: 1.0,    # Routine visits less important
        }
    """
    def __init__(self, vocab_size: int, category_weights: dict = None, default_weight: float = 1.0):
        super().__init__()
        
        # Initialize all weights to default
        weights = torch.full((vocab_size,), default_weight, dtype=torch.float32)
        
        # Override with custom weights
        if category_weights:
            for token_id, weight in category_weights.items():
                weights[token_id] = weight
        
        self.register_buffer('token_weights', weights)
    
    def forward(self, token_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) - token IDs from input_ids[:, 0, :]
            padding_mask: (batch, seq_len) - 1 for real tokens, 0 for padding
        
        Returns:
            weights: (batch, seq_len) - importance weights
        """
        # Gather weights for each token
        weights = self.token_weights[token_ids]  # (batch, seq_len)
        
        # Zero out padding positions
        weights = weights * padding_mask.float()
        
        # Normalize to sum to 1 per sequence (excluding padding)
        sum_weights = weights.sum(dim=1, keepdim=True)
        weights = weights / (sum_weights + 1e-8)
        
        return weights


class CombinedImportanceWeighting(nn.Module):
    """
    Combine multiple weighting strategies:
    - Temporal recency
    - Category importance
    - Custom attention bias
    
    Args:
        temporal_decay: Decay rate for temporal weighting
        vocab_size: Vocabulary size
        category_weights: Dict of token_id -> weight
        combination: How to combine weights ('multiply', 'add', 'learned')
    """
    def __init__(
        self,
        temporal_decay: float = 0.1,
        vocab_size: int = None,
        category_weights: dict = None,
        combination: str = 'multiply'
    ):
        super().__init__()
        
        self.combination = combination
        
        # Temporal component
        self.temporal = TemporalImportanceWeighting(decay_rate=temporal_decay, learnable=False)
        
        # Category component
        if vocab_size and category_weights:
            self.category = CategoryBasedWeighting(vocab_size, category_weights)
        else:
            self.category = None
        
        # Learned combination weights
        if combination == 'learned':
            self.alpha = nn.Parameter(torch.tensor(0.5))  # balance between temporal and category
    
    def forward(
        self,
        token_ids: torch.Tensor,
        ages: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len)
            ages: (batch, seq_len)
            padding_mask: (batch, seq_len)
        
        Returns:
            combined_weights: (batch, seq_len)
        """
        # Get temporal weights
        temporal_weights = self.temporal(ages)
        
        # Get category weights if available
        if self.category:
            category_weights = self.category(token_ids, padding_mask)
            
            # Combine
            if self.combination == 'multiply':
                weights = temporal_weights * category_weights
            elif self.combination == 'add':
                weights = (temporal_weights + category_weights) / 2
            elif self.combination == 'learned':
                weights = self.alpha * temporal_weights + (1 - self.alpha) * category_weights
            else:
                raise ValueError(f"Unknown combination: {self.combination}")
        else:
            weights = temporal_weights
        
        # Apply padding mask
        weights = weights * padding_mask.float()
        
        # Normalize
        sum_weights = weights.sum(dim=1, keepdim=True)
        weights = weights / (sum_weights + 1e-8)
        
        return weights


class WeightedPooling(nn.Module):
    """
    Weighted pooling for sequence embeddings.
    Instead of mean/CLS pooling, use importance-weighted average.
    
    Args:
        hidden_size: Embedding dimension
        weighting_strategy: 'learned', 'temporal', 'category', or 'combined'
        **kwargs: Arguments for the weighting module
    """
    def __init__(
        self,
        hidden_size: int,
        weighting_strategy: str = 'learned',
        **kwargs
    ):
        super().__init__()
        
        self.strategy = weighting_strategy
        
        if weighting_strategy == 'learned':
            # Learn attention weights from embeddings
            self.query = nn.Linear(hidden_size, 1)
        elif weighting_strategy == 'temporal':
            self.weighting = TemporalImportanceWeighting(**kwargs)
        elif weighting_strategy == 'category':
            self.weighting = CategoryBasedWeighting(**kwargs)
        elif weighting_strategy == 'combined':
            self.weighting = CombinedImportanceWeighting(**kwargs)
        else:
            raise ValueError(f"Unknown strategy: {weighting_strategy}")
    
    def forward(
        self,
        embeddings: torch.Tensor,
        padding_mask: torch.Tensor,
        token_ids: torch.Tensor = None,
        ages: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, seq_len, hidden_dim)
            padding_mask: (batch, seq_len)
            token_ids: (batch, seq_len) - needed for category weighting
            ages: (batch, seq_len) - needed for temporal weighting
        
        Returns:
            pooled: (batch, hidden_dim)
        """
        if self.strategy == 'learned':
            # Compute attention scores from embeddings
            scores = self.query(embeddings).squeeze(-1)  # (batch, seq_len)
            
            # Mask padding
            scores = scores.masked_fill(padding_mask == 0, float('-inf'))
            
            # Softmax to get weights
            weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        else:
            # Use pre-defined weighting strategy
            weights = self.weighting(token_ids, ages, padding_mask)
            weights = weights.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Weighted sum
        pooled = (embeddings * weights).sum(dim=1)  # (batch, hidden_dim)
        
        return pooled


# ─────────────────────────────────────────────────────────────────────
# Example Usage Functions
# ─────────────────────────────────────────────────────────────────────

def create_importance_weights_from_vocab(vocab_path: str, importance_col: str = 'IMPORTANCE'):
    """
    Helper to create category_weights dict from vocab CSV.
    
    Args:
        vocab_path: Path to vocab.csv
        importance_col: Column name containing importance scores
    
    Returns:
        category_weights: Dict mapping token_id -> weight
    """
    import pandas as pd
    
    df = pd.read_csv(vocab_path)
    
    # If no importance column, return None
    if importance_col not in df.columns:
        return None
    
    # Create mapping
    category_weights = {}
    for _, row in df.iterrows():
        token_id = row['ID']
        importance = row[importance_col]
        if pd.notna(importance) and importance > 0:
            category_weights[token_id] = float(importance)
    
    return category_weights


def add_weighted_pooling_to_finetune(
    encoder_output: torch.Tensor,
    input_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    weighting_module: WeightedPooling
) -> torch.Tensor:
    """
    Example of using weighted pooling in fine-tuning forward pass.
    
    Args:
        encoder_output: (batch, seq_len, hidden_dim) from transformer
        input_ids: (batch, 4, seq_len) original inputs
        padding_mask: (batch, seq_len)
        weighting_module: Instance of WeightedPooling
    
    Returns:
        pooled_embedding: (batch, hidden_dim)
    """
    token_ids = input_ids[:, 0, :]  # Token dimension
    ages = input_ids[:, 2, :]        # Age dimension
    
    pooled = weighting_module(
        embeddings=encoder_output,
        padding_mask=padding_mask,
        token_ids=token_ids,
        ages=ages
    )
    
    return pooled
