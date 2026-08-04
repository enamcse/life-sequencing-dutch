"""
PRACTICAL IMPLEMENTATION: Weighted Attention for Your Fine-Tuning

This file shows the EXACT code changes to add to your existing files.
Copy-paste these modifications into your codebase.
"""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Modify finetune_model.py to support weighted pooling
# ═══════════════════════════════════════════════════════════════════════════

# Add this import at the top of finetune_model.py (around line 10-20)
"""
from pop2vec.llm.src.transformer.weighted_attention_addon import WeightedPooling
"""

# Modify the _init_decoder method (around line 119-140):

"""
BEFORE (your current code):
```python
def _init_decoder(self) -> None:
    self.num_outputs = self.hparams["num_targets"]

    if self.hparams["pooled"]:
        self.decoder = AttentionDecoder(self.hparams, num_outputs=self.num_outputs)
    else:
        self.decoder = CLS_DecoderS(self.hparams)
```

AFTER (modified code with weighted pooling option):
```python
def _init_decoder(self) -> None:
    self.num_outputs = self.hparams["num_targets"]

    if self.hparams["pooled"]:
        # Check if we want weighted pooling
        if self.hparams.get("use_weighted_pooling", False):
            # Use importance-weighted pooling instead of AttentionDecoder
            self.weighted_pooling = WeightedPooling(
                hidden_size=self.hparams["hidden_size"],
                weighting_strategy=self.hparams.get("weighting_strategy", "learned")
            )
            # Simple linear head after weighted pooling
            self.decoder = nn.Linear(self.hparams["hidden_size"], self.num_outputs)
        else:
            # Original: Use AttentionDecoder (which has its own attention pooling)
            self.decoder = AttentionDecoder(self.hparams, num_outputs=self.num_outputs)
    else:
        # CLS token only approach
        self.decoder = CLS_DecoderS(self.hparams)
```
"""

# Modify the forward method (around line 186-202):

"""
BEFORE (your current code):
```python
def forward(self, batch: Dict[str, torch.Tensor], invert=False) -> torch.Tensor:
    hidden = self.encoder_forward(
        x=batch["input_ids"].long(),
        padding_mask=batch["padding_mask"].long(),
    )
    if self.hparams["pooled"]:
        out = self.decoder(hidden, mask=batch["padding_mask"].long())
    else:
        out = self.decoder(hidden)
    if invert:
        out = out * self.sigma + self.mu
    return out
```

AFTER (modified with weighted pooling support):
```python
def forward(self, batch: Dict[str, torch.Tensor], invert=False) -> torch.Tensor:
    hidden = self.encoder_forward(
        x=batch["input_ids"].long(),
        padding_mask=batch["padding_mask"].long(),
    )
    if self.hparams["pooled"]:
        if self.hparams.get("use_weighted_pooling", False):
            # NEW: Use weighted pooling
            pooled = self.weighted_pooling(
                embeddings=hidden,
                padding_mask=batch["padding_mask"].long(),
                token_ids=batch["input_ids"][:, 0, :].long(),
                ages=batch["input_ids"][:, 2, :].long()
            )
            out = self.decoder(pooled)
        else:
            # Original: AttentionDecoder with its own pooling
            out = self.decoder(hidden, mask=batch["padding_mask"].long())
    else:
        out = self.decoder(hidden)
    if invert:
        out = out * self.sigma + self.mu
    return out
```
"""


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Update your hyperparameters file
# ═══════════════════════════════════════════════════════════════════════════

"""
Add these lines to your fine-tuning config (e.g., finetune_hparams.txt):

# Weighted Attention Settings (NEW)
use_weighted_pooling: False   # Set to True to enable
weighting_strategy: "learned"  # Options: learned, temporal, category, combined

# If using temporal or combined strategy:
temporal_decay: 0.1

# If using category-based weighting, optionally add IMPORTANCE column to vocab.csv
"""


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: How to test/compare
# ═══════════════════════════════════════════════════════════════════════════

"""
BASELINE RUN (your current setup):
```bash
python -m pop2vec.llm.src.new_code.finetune_new --hparams configs/finetune_baseline.txt
```

Content of configs/finetune_baseline.txt:
```
# ... all your normal settings ...
use_weighted_pooling: False   # Use default AttentionDecoder
```

WEIGHTED POOLING RUN (test learned weighting):
```bash
python -m pop2vec.llm.src.new_code.finetune_new --hparams configs/finetune_weighted.txt
```

Content of configs/finetune_weighted.txt:
```
# ... all your normal settings ...
use_weighted_pooling: True
weighting_strategy: "learned"   # Learns importance from data
```

TEMPORAL WEIGHTING RUN (recent events matter more):
```bash
python -m pop2vec.llm.src.new_code.finetune_new --hparams configs/finetune_temporal.txt
```

Content of configs/finetune_temporal.txt:
```
# ... all your normal settings ...
use_weighted_pooling: True
weighting_strategy: "temporal"
temporal_decay: 0.1   # Higher = stronger recency bias (try 0.05, 0.1, 0.2)
```
"""


# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON: What Each Approach Does
# ═══════════════════════════════════════════════════════════════════════════

"""
YOUR CURRENT SETUP (What you already have):
┌─────────────────────────────────────────────────────────────┐
│ 1. Class-Imbalanced Loss Weighting                         │
│    WHERE: CrossEntropyLoss(weight=...)                     │
│    WHAT: Minority classes get higher loss weight           │
│    WHEN TO USE: Always good for imbalanced datasets        │
│                                                             │
│ 2. Sample Weighting (WeightedRandomSampler)                │
│    WHERE: DataLoader with custom sampler                   │
│    WHAT: Oversamples minority class examples               │
│    WHEN TO USE: When you have oversample: True in config   │
│                                                             │
│ 3. Padding Mask                                            │
│    WHERE: Attention mechanism                              │
│    WHAT: Ignores padding tokens completely                 │
│    WHEN TO USE: Always (automatic)                         │
└─────────────────────────────────────────────────────────────┘

NEW OPTIONS (What you can add):
┌─────────────────────────────────────────────────────────────┐
│ 4. Learned Weighted Pooling                                │
│    WHERE: Between encoder and decoder (pooling step)       │
│    WHAT: Neural network learns which tokens are important  │
│    WHEN TO USE: When you don't have domain knowledge       │
│    HOW: weighting_strategy: "learned"                      │
│                                                             │
│ 5. Temporal Weighted Pooling                               │
│    WHERE: Between encoder and decoder (pooling step)       │
│    WHAT: Recent events get higher weight                   │
│    WHEN TO USE: When recent history matters more           │
│    HOW: weighting_strategy: "temporal", temporal_decay: 0.1│
│                                                             │
│ 6. Category-Based Weighted Pooling                         │
│    WHERE: Between encoder and decoder (pooling step)       │
│    WHAT: Manually assign importance to token types         │
│    WHEN TO USE: When you have domain expertise             │
│    HOW: weighting_strategy: "category" + add IMPORTANCE    │
│         column to vocab.csv                                │
│                                                             │
│ 7. Combined Weighted Pooling                               │
│    WHERE: Between encoder and decoder (pooling step)       │
│    WHAT: Combines temporal + category weighting            │
│    WHEN TO USE: For maximum control                        │
│    HOW: weighting_strategy: "combined"                     │
└─────────────────────────────────────────────────────────────┘

VISUALIZATION OF DATA FLOW:

Without weighted pooling (current):
    Input Sequence → Transformer Encoder → AttentionDecoder (pooling) → Linear → Output

With weighted pooling (new):
    Input Sequence → Transformer Encoder → WeightedPooling → Linear → Output
                                               ↑
                                     Uses token importance
"""


# ═══════════════════════════════════════════════════════════════════════════
# PRACTICAL EXAMPLE: Adding IMPORTANCE to vocab.csv
# ═══════════════════════════════════════════════════════════════════════════

"""
If you want to use category-based weighting, edit your vocab.csv to add an IMPORTANCE column:

BEFORE:
```csv
ID,TOKEN,CATEGORY
0,[PAD],SPECIAL
1,[CLS],SPECIAL
2,[DEATH],VITAL_EVENT
3,HOSPITAL_ADMISSION,MEDICAL_EVENT
4,ROUTINE_CHECKUP,MEDICAL_EVENT
5,CANCER_DIAGNOSIS,DIAGNOSIS
...
```

AFTER:
```csv
ID,TOKEN,CATEGORY,IMPORTANCE
0,[PAD],SPECIAL,0.0
1,[CLS],SPECIAL,1.0
2,[DEATH],VITAL_EVENT,10.0
3,HOSPITAL_ADMISSION,MEDICAL_EVENT,5.0
4,ROUTINE_CHECKUP,MEDICAL_EVENT,1.0
5,CANCER_DIAGNOSIS,DIAGNOSIS,8.0
...
```

Rules of thumb for IMPORTANCE values:
- 0.0: Ignore completely (padding)
- 1.0: Normal importance (routine events)
- 2.0-3.0: Moderately important (minor medical events)
- 5.0-7.0: Important (hospitalizations, significant diagnoses)
- 8.0-10.0: Critical (death, major diagnoses, life-changing events)

Then set in config:
```
use_weighted_pooling: True
weighting_strategy: "category"
```
"""


# ═══════════════════════════════════════════════════════════════════════════
# DEBUGGING: How to check if it's working
# ═══════════════════════════════════════════════════════════════════════════

"""
Add this to your training_step or validation_step in finetune_model.py
to see the learned weights:

```python
def validation_step(self, batch, batch_idx):
    # ... your existing code ...
    
    # Debug weighted pooling (optional)
    if (batch_idx == 0 and 
        self.hparams.get("use_weighted_pooling", False) and 
        hasattr(self, 'weighted_pooling')):
        
        # Get attention weights from the pooling module
        with torch.no_grad():
            hidden = self.encoder_forward(
                x=batch["input_ids"].long(),
                padding_mask=batch["padding_mask"].long(),
            )
            
            # Extract weights for first sample
            token_ids = batch["input_ids"][0, 0, :].long()
            ages = batch["input_ids"][0, 2, :].long()
            padding_mask = batch["padding_mask"][0].long()
            
            # Get the attention weights
            if self.hparams["weighting_strategy"] == "learned":
                scores = self.weighted_pooling.query(hidden[0])
                weights = torch.softmax(scores.squeeze(), dim=0)
            # ... handle other strategies ...
            
            # Log top 10 most important tokens
            top_k = 10
            top_weights, top_indices = weights.topk(top_k)
            logger.info(f"Top {top_k} weighted tokens:")
            for i, (w, idx) in enumerate(zip(top_weights, top_indices)):
                tok_id = token_ids[idx].item()
                age = ages[idx].item()
                logger.info(f"  {i+1}. Token {tok_id} at age {age}: weight={w:.4f}")
```
"""


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY: YOUR ACTION ITEMS
# ═══════════════════════════════════════════════════════════════════════════

"""
IMMEDIATE NEXT STEPS:

1. ✅ You already have weighted_attention_addon.py created
2. ✅ You already have the integration guide

3. TODO: Modify finetune_model.py (5 minutes)
   - Add import for WeightedPooling
   - Modify _init_decoder() method
   - Modify forward() method
   
4. TODO: Create test config (2 minutes)
   - Copy your existing finetune config
   - Add: use_weighted_pooling: True
   - Add: weighting_strategy: "learned"
   
5. TODO: Run comparison (overnight)
   - Run baseline without weighted pooling
   - Run with weighted pooling
   - Compare AUC, F1, accuracy

6. OPTIONAL: If results are good
   - Try temporal weighting
   - Try category weighting with manual IMPORTANCE scores
   - Try combined weighting

EXPECTED IMPACT:
- Learned weighting: 0-5% improvement (learns from data)
- Temporal weighting: 0-10% improvement if recency matters
- Category weighting: 5-15% improvement with good domain knowledge
- Combined: Best of both worlds

Remember: This is ORTHOGONAL to your existing sample weighting!
You can (and should) use both together.
"""
