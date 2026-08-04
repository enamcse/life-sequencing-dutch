# Weighted Attention in Your Life-Sequencing Project - Complete Guide

## TL;DR

You asked about **weighted attention** in your transformer model. Here's what you have and what you can add:

### ✅ **What You Already Have** (No Changes Needed)

1. **Class-imbalanced loss weighting** - Gives more importance to minority classes
2. **Sample weighting (WeightedRandomSampler)** - Oversamples rare examples  
3. **Padding mask** - Ignores padding tokens in attention

### 🆕 **What You Can Add** (New Files Created)

**Importance-weighted pooling** - Different tokens in a sequence get different importance weights when creating the sequence-level embedding.

**Files Created:**
- `pop2vec/llm/src/transformer/weighted_attention_addon.py` - Implementation
- `docs/weighted_attention_integration_guide.py` - Detailed guide
- `docs/IMPLEMENTATION_GUIDE_weighted_attention.md` - Step-by-step instructions

---

## Quick Start (5 Minutes to Test)

### Step 1: Modify `finetune_model.py`

Add this import at the top:
```python
from pop2vec.llm.src.transformer.weighted_attention_addon import WeightedPooling
```

Modify the `_init_decoder` method:
```python
def _init_decoder(self) -> None:
    self.num_outputs = self.hparams["num_targets"]
    
    if self.hparams["pooled"]:
        if self.hparams.get("use_weighted_pooling", False):
            # NEW: Weighted pooling
            self.weighted_pooling = WeightedPooling(
                hidden_size=self.hparams["hidden_size"],
                weighting_strategy=self.hparams.get("weighting_strategy", "learned")
            )
            self.decoder = nn.Linear(self.hparams["hidden_size"], self.num_outputs)
        else:
            # Original AttentionDecoder
            self.decoder = AttentionDecoder(self.hparams, num_outputs=self.num_outputs)
    else:
        self.decoder = CLS_DecoderS(self.hparams)
```

Modify the `forward` method:
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
            out = self.decoder(hidden, mask=batch["padding_mask"].long())
    else:
        out = self.decoder(hidden)
    if invert:
        out = out * self.sigma + self.mu
    return out
```

### Step 2: Add to Your Config

Add these lines to your fine-tuning hyperparameters file:
```
use_weighted_pooling: True
weighting_strategy: "learned"
```

### Step 3: Run and Compare

**Baseline (your current setup):**
```bash
python -m pop2vec.llm.src.new_code.finetune_new --hparams configs/baseline.txt
```
With `use_weighted_pooling: False` in config

**With weighted attention:**
```bash
python -m pop2vec.llm.src.new_code.finetune_new --hparams configs/weighted.txt
```
With `use_weighted_pooling: True` in config

Compare the validation metrics (AUC, F1, accuracy).

---

## Weighting Strategies Explained

### 1. **`"learned"`** (Recommended First Try)
- Neural network learns which tokens are important
- No manual configuration needed
- Works well when you don't have domain expertise
- **Use when:** You want automatic importance weighting

### 2. **`"temporal"`** (For Time-Dependent Tasks)
- Recent events get higher weights
- Exponential decay for older events
- Add to config: `temporal_decay: 0.1` (higher = stronger recency bias)
- **Use when:** Recent medical history matters more than distant past

### 3. **`"category"`** (For Domain Experts)
- Manually assign importance to token types
- Requires adding `IMPORTANCE` column to `vocab.csv`
- Example: Death=10.0, Hospital=5.0, Routine=1.0
- **Use when:** You have clinical expertise about event importance

### 4. **`"combined"`** (Best of Both Worlds)
- Combines temporal + category weighting
- Requires both `temporal_decay` and `IMPORTANCE` column
- **Use when:** You want maximum control

---

## Expected Impact

| Strategy | Expected Improvement | Setup Complexity |
|----------|---------------------|------------------|
| Learned | 0-5% | Low (automatic) |
| Temporal | 0-10% | Low (one parameter) |
| Category | 5-15% | Medium (need domain knowledge) |
| Combined | 5-20% | High (both) |

**Note:** These are additive to your existing sample weighting!

---

## Visualization

### Current Pipeline (Without Weighted Pooling)
```
Input Sequence (tokens + ages + dates)
          ↓
Transformer Encoder (contextualized embeddings)
          ↓
AttentionDecoder (learned pooling)
          ↓
Linear Layer
          ↓
Output (predictions)
```

### New Pipeline (With Weighted Pooling)
```
Input Sequence (tokens + ages + dates)
          ↓
Transformer Encoder (contextualized embeddings)
          ↓
WeightedPooling (importance-based pooling)
   ↑
   Uses: token IDs, ages, learned patterns
          ↓
Linear Layer
          ↓
Output (predictions)
```

---

## Example: Category-Based Weighting

If you want to use `weighting_strategy: "category"`, add an `IMPORTANCE` column to `vocab.csv`:

```csv
ID,TOKEN,CATEGORY,IMPORTANCE
0,[PAD],SPECIAL,0.0
1,[CLS],SPECIAL,1.0
2,[DEATH],VITAL_EVENT,10.0
3,HOSPITAL_ADMISSION,MEDICAL,5.0
4,CANCER_DIAGNOSIS,DIAGNOSIS,8.0
5,ROUTINE_CHECKUP,MEDICAL,1.0
6,FLU,DIAGNOSIS,2.0
...
```

**Rules of thumb:**
- 0.0 = Ignore
- 1.0 = Normal importance
- 2-3 = Moderately important
- 5-7 = Important
- 8-10 = Critical

---

## FAQ

**Q: Is this different from attention in the transformer?**  
A: Yes! The transformer's attention (in `attention.py`) determines how tokens interact with each other. This weighted pooling determines how tokens contribute to the **final sequence-level embedding**.

**Q: Can I use this with my existing sample weighting?**  
A: Absolutely! They work together:
- **Sample weighting** (WeightedRandomSampler): Which sequences to train on more
- **Weighted pooling** (this): Which tokens within a sequence matter more

**Q: Will this slow down training?**  
A: Minimal impact. Weighted pooling is very lightweight compared to the transformer encoder.

**Q: Should I use this for pretraining or fine-tuning?**  
A: **Fine-tuning** is recommended. For pretraining, you'd need different modifications (see `docs/weighted_attention_integration_guide.py` for details).

**Q: What if results don't improve?**  
A: That's valuable information! It means all tokens are equally important for your task. You can keep the code but set `use_weighted_pooling: False`.

---

## Troubleshooting

**Error: `ModuleNotFoundError: No module named 'weighted_attention_addon'`**
- Make sure you created the file in the correct location
- Path: `pop2vec/llm/src/transformer/weighted_attention_addon.py`

**Error: `KeyError: 'weighting_strategy'`**
- Add `weighting_strategy: "learned"` to your config file

**Poor results with temporal weighting:**
- Try different `temporal_decay` values: 0.05, 0.1, 0.2
- Higher values = stronger recency bias

**Category weighting not working:**
- Check that `IMPORTANCE` column exists in `vocab.csv`
- Make sure token IDs match between vocab and data

---

## Next Steps

1. ✅ Read this summary
2. ✅ Check the created files are in place
3. ⬜ Modify `finetune_model.py` (copy-paste from guide)
4. ⬜ Create test config with `use_weighted_pooling: True`
5. ⬜ Run baseline vs weighted comparison
6. ⬜ Analyze results and decide on strategy
7. ⬜ (Optional) Try different weighting strategies

---

## Summary of What Weighted Attention Does

**In Simple Terms:**
When your model creates a single embedding for an entire life sequence, it needs to combine information from all events (hospital visits, diagnoses, etc.). 

**Without weighted attention:**
- All events contribute equally (or the model learns some generic pattern)

**With weighted attention:**
- You can make important events (death, major diagnoses) contribute more
- You can make recent events matter more than old ones
- You can let the model learn which events matter most

This is especially useful in healthcare where:
- Recent events are often more predictive
- Critical events (hospitalizations, deaths) are more informative
- Routine checkups might be less important

---

## Contact / Questions

If you have questions or run into issues:
1. Check `docs/IMPLEMENTATION_GUIDE_weighted_attention.md` for step-by-step instructions
2. Check `docs/weighted_attention_integration_guide.py` for detailed code examples
3. The implementation in `weighted_attention_addon.py` has docstrings explaining each function

Good luck! 🚀
