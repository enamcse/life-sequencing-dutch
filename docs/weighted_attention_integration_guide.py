"""
Guide: How to Use Weighted Attention in Your Life-Sequencing Project

This document explains where and how to integrate weighted attention mechanisms
into your existing codebase.
"""

# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 1: Weighted Pooling in Fine-Tuning (RECOMMENDED - EASIEST)
# ═══════════════════════════════════════════════════════════════════════════

"""
This approach adds importance-weighted pooling to your fine-tuning decoder,
replacing the simple mean/CLS pooling.

WHERE TO CHANGE: finetune_model.py in the _init_decoder() method
"""

# In pop2vec/llm/src/new_code/finetune_model.py

from pop2vec.llm.src.transformer.weighted_attention_addon import (
    WeightedPooling,
    create_importance_weights_from_vocab
)

class TransformerFT(pl.LightningModule):
    
    def _init_decoder(self) -> None:
        """Modified version with weighted pooling option"""
        self.num_outputs = self.hparams["num_targets"]

        if self.hparams["pooled"]:
            # Option A: Use existing AttentionDecoder (already does learned weighting)
            if self.hparams.get("use_weighted_pooling", False):
                # NEW: Use importance-weighted pooling
                weighting_strategy = self.hparams.get("weighting_strategy", "learned")
                
                if weighting_strategy == "category":
                    # Load category weights from vocab
                    category_weights = create_importance_weights_from_vocab(
                        self.hparams["vocab_path"],
                        importance_col="IMPORTANCE"  # Add this column to vocab.csv if needed
                    )
                    self.pooling = WeightedPooling(
                        hidden_size=self.hparams["hidden_size"],
                        weighting_strategy="category",
                        vocab_size=self.hparams["vocab_size"],
                        category_weights=category_weights
                    )
                elif weighting_strategy == "temporal":
                    self.pooling = WeightedPooling(
                        hidden_size=self.hparams["hidden_size"],
                        weighting_strategy="temporal",
                        decay_rate=self.hparams.get("temporal_decay", 0.1)
                    )
                elif weighting_strategy == "combined":
                    category_weights = create_importance_weights_from_vocab(
                        self.hparams["vocab_path"],
                        importance_col="IMPORTANCE"
                    )
                    self.pooling = WeightedPooling(
                        hidden_size=self.hparams["hidden_size"],
                        weighting_strategy="combined",
                        temporal_decay=self.hparams.get("temporal_decay", 0.1),
                        vocab_size=self.hparams["vocab_size"],
                        category_weights=category_weights,
                        combination=self.hparams.get("combination", "multiply")
                    )
                else:  # "learned"
                    self.pooling = WeightedPooling(
                        hidden_size=self.hparams["hidden_size"],
                        weighting_strategy="learned"
                    )
                
                # Simple decoder head after pooling
                self.decoder = nn.Linear(self.hparams["hidden_size"], self.num_outputs)
            else:
                # Original: Use AttentionDecoder (from transformer.py)
                from pop2vec.llm.src.transformer.transformer import AttentionDecoder
                self.decoder = AttentionDecoder(self.hparams, num_outputs=self.num_outputs)
        else:
            # CLS token only - no pooling needed
            from pop2vec.llm.src.transformer.transformer import CLS_DecoderS
            self.decoder = CLS_DecoderS(self.hparams)

    def forward(self, batch: Dict[str, torch.Tensor], invert=False) -> torch.Tensor:
        """Modified forward with weighted pooling support"""
        hidden = self.encoder_forward(
            x=batch["input_ids"].long(),
            padding_mask=batch["padding_mask"].long(),
        )
        
        if self.hparams["pooled"]:
            if self.hparams.get("use_weighted_pooling", False):
                # NEW: Use weighted pooling
                pooled = self.pooling(
                    embeddings=hidden,
                    padding_mask=batch["padding_mask"].long(),
                    token_ids=batch["input_ids"][:, 0, :].long(),
                    ages=batch["input_ids"][:, 2, :].long()
                )
                out = self.decoder(pooled)
            else:
                # Original: Use AttentionDecoder which has its own pooling
                out = self.decoder(hidden, mask=batch["padding_mask"].long())
        else:
            # CLS token approach
            out = self.decoder(hidden)
        
        if invert:
            out = out * self.sigma + self.mu
        return out


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 2: Token-Level Weighting in Loss Calculation
# ═══════════════════════════════════════════════════════════════════════════

"""
This approach weights individual tokens differently in the loss calculation
during pretraining (MLM or AR-LM).

WHERE TO CHANGE: models.py in the training_step() method
"""

# In pop2vec/llm/src/transformer/models.py

from pop2vec.llm.src.transformer.weighted_attention_addon import (
    TemporalImportanceWeighting,
    CategoryBasedWeighting
)

class TransformerEncoder(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        # ... existing code ...
        
        # Add weighting modules if enabled
        if hparams.get("use_token_weighting", False):
            self.token_weighting = CategoryBasedWeighting(
                vocab_size=hparams.vocab_size,
                category_weights=hparams.get("token_importance_weights", None)
            )
            self.temporal_weighting = TemporalImportanceWeighting(
                decay_rate=hparams.get("temporal_decay", 0.1)
            )
        else:
            self.token_weighting = None
            self.temporal_weighting = None
    
    def training_step(self, batch, batch_idx):
        """Modified training step with token-level weighting"""
        if self.task == "ar_lm":
            # ... existing AR-LM code ...
            loss = F.cross_entropy(
                predictions.view(-1, self.num_outputs),
                targets.view(-1),
                ignore_index=0,
                reduction='none'  # Get per-token losses
            )
            
            # Apply token-level weighting if enabled
            if self.token_weighting is not None:
                token_ids = batch["input_ids"][:, 0, :]
                ages = batch["input_ids"][:, 2, :]
                padding_mask = batch["padding_mask"]
                
                # Get importance weights
                token_weights = self.token_weighting(token_ids, padding_mask)
                temporal_weights = self.temporal_weighting(ages)
                combined_weights = token_weights * temporal_weights
                
                # Apply weights to loss
                loss = loss.view(predictions.size(0), -1)
                weighted_loss = (loss * combined_weights).sum() / combined_weights.sum()
            else:
                # Original: simple mean
                weighted_loss = loss.mean()
            
            # ... rest of training step ...


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 3: Modify Attention Mechanism Directly (ADVANCED - NOT RECOMMENDED)
# ═══════════════════════════════════════════════════════════════════════════

"""
This would modify the attention scores in the Performer attention mechanism.
NOT RECOMMENDED because:
1. Your Performer attention already handles local vs global heads
2. It's complex to modify the FAVOR+ algorithm
3. The above approaches are simpler and more effective

If you really want this, you'd need to modify:
- pop2vec/llm/src/transformer/performer.py
- Add bias terms to attention scores before softmax
"""


# ═══════════════════════════════════════════════════════════════════════════
# HOW TO ENABLE IN YOUR CONFIG FILES
# ═══════════════════════════════════════════════════════════════════════════

"""
For fine-tuning, add to your hparams file (e.g., regular_hparams_small.txt):

# Weighted attention settings
use_weighted_pooling: True
weighting_strategy: "combined"  # Options: learned, temporal, category, combined
temporal_decay: 0.1              # How quickly old events lose importance
combination: "multiply"          # How to combine temporal + category (multiply/add/learned)

For pretraining with token-level weighting:

use_token_weighting: True
temporal_decay: 0.1
"""

# Example config for fine-tuning:
FINETUNE_CONFIG_EXAMPLE = {
    "sequence_encoded": "/path/to/sequences.h5",
    "label_file": "/path/to/labels.csv",
    "checkpoint_path": "/path/to/pretrained.ckpt",
    "vocab_path": "/path/to/vocab.csv",
    
    # Standard settings
    "batch_size": 64,
    "num_targets": 2,
    "task_type": "binary",
    "pooled": True,
    
    # NEW: Weighted attention settings
    "use_weighted_pooling": True,
    "weighting_strategy": "combined",  # learned/temporal/category/combined
    "temporal_decay": 0.1,
    "combination": "multiply",
    
    # ... other settings ...
}


# ═══════════════════════════════════════════════════════════════════════════
# OPTIONAL: ADD IMPORTANCE SCORES TO VOCAB.CSV
# ═══════════════════════════════════════════════════════════════════════════

"""
To use category-based weighting, add an IMPORTANCE column to your vocab.csv:

ID,TOKEN,CATEGORY,IMPORTANCE
0,[PAD],SPECIAL,0.0
1,[CLS],SPECIAL,1.0
2,[DEATH],VITAL,10.0
3,HOSPITAL_ADMISSION,MEDICAL,5.0
4,ROUTINE_CHECKUP,MEDICAL,1.0
5,CANCER_DIAGNOSIS,MEDICAL,8.0
...

Higher values = more important tokens
"""


# ═══════════════════════════════════════════════════════════════════════════
# TESTING THE NEW FUNCTIONALITY
# ═══════════════════════════════════════════════════════════════════════════

"""
To test weighted pooling without full training:
"""

def test_weighted_pooling():
    import torch
    from pop2vec.llm.src.transformer.weighted_attention_addon import WeightedPooling
    
    # Create dummy data
    batch_size = 4
    seq_len = 100
    hidden_dim = 512
    vocab_size = 1000
    
    embeddings = torch.randn(batch_size, seq_len, hidden_dim)
    padding_mask = torch.ones(batch_size, seq_len)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    ages = torch.linspace(0, 100, seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Test learned weighting
    pooling = WeightedPooling(hidden_dim, weighting_strategy="learned")
    pooled = pooling(embeddings, padding_mask, token_ids, ages)
    print(f"Learned pooling output shape: {pooled.shape}")  # Should be (4, 512)
    
    # Test temporal weighting
    pooling = WeightedPooling(hidden_dim, weighting_strategy="temporal", decay_rate=0.1)
    pooled = pooling(embeddings, padding_mask, token_ids, ages)
    print(f"Temporal pooling output shape: {pooled.shape}")
    
    # Test category weighting
    category_weights = {100: 10.0, 200: 5.0}  # Some important tokens
    pooling = WeightedPooling(
        hidden_dim, 
        weighting_strategy="category",
        vocab_size=vocab_size,
        category_weights=category_weights
    )
    pooled = pooling(embeddings, padding_mask, token_ids, ages)
    print(f"Category pooling output shape: {pooled.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY: WHAT TO DO NEXT
# ═══════════════════════════════════════════════════════════════════════════

"""
RECOMMENDED STEPS:

1. START SIMPLE - Use Approach 1 (Weighted Pooling in Fine-Tuning):
   - Add the code modifications to finetune_model.py
   - Set use_weighted_pooling: True in your config
   - Try weighting_strategy: "learned" first (no extra setup needed)
   - Compare results with baseline

2. IF STEP 1 HELPS - Try domain-specific strategies:
   - Add IMPORTANCE column to vocab.csv with clinical knowledge
   - Try weighting_strategy: "category" to use those weights
   - Try weighting_strategy: "temporal" for recency bias
   - Try weighting_strategy: "combined" for both

3. ADVANCED - If you need token-level weighting in pretraining:
   - Use Approach 2 (modify models.py)
   - This is more complex and may not help much

4. MEASURE IMPACT:
   - Log both weighted and unweighted metrics
   - Check if minority class performance improves
   - Check if recency bias makes sense for your task

You already have sample weighting (WeightedRandomSampler) working,
so the main new capability is importance-weighted pooling!
"""
