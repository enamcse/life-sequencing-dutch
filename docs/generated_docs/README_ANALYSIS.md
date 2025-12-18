# Generative Model Analysis and Evaluation Scripts

This directory contains scripts for analyzing and evaluating generative life sequence models.

## Overview

The scripts are organized to help you:
1. **Fix the output format bug** in `utils.py`
2. **Repair malformed output files** from previous runs
3. **Analyze generated sequences** for statistics and patterns
4. **Evaluate generative models** against real data

## Files

### 1. `fix_pretty_print_tokens.py`
Reference implementation showing the corrected version of `pretty_print_tokens` function.

**What it fixes:** The original function used `max_per_line=20`, causing long sequences (horizon > 20 or prefix_len > 20) to be split across multiple lines.

**Status:** ✅ **Already applied to `utils.py`**

### 2. `repair_generative_output.py`
Repairs malformed output files by:
- Merging continuation lines (when sequences were split)
- Removing duplicates (from parallel GPU execution)
- Ensuring proper alternating pattern (ORIGINAL, GENERATED, ORIGINAL, GENERATED, ...)

**Usage:**
```bash
python repair_generative_output.py \
    /projects/0/prjs1589/stonybrook/llm/gen_out/pretty_tokens_20251212.txt \
    /projects/0/prjs1589/stonybrook/llm/gen_out/pretty_tokens_20251212_repaired.txt
```

### 3. `analyze_generative_output.py`
Comprehensive analysis of generated sequences including:
- Category distribution of generated tokens
- Token match rate (exact matches with continuation)
- Category match rate
- Diversity metrics (unique tokens, repetition rates)
- Per-sequence statistics

**Usage:**
```bash
python analyze_generative_output.py \
    /projects/0/prjs1589/stonybrook/llm/gen_out/pretty_tokens_20251212_repaired.txt \
    --output_dir /projects/0/prjs1589/stonybrook/llm/gen_out/analysis/
```

**Outputs:**
- `summary.json`: Overall statistics
- `per_sequence_stats.csv`: Detailed per-sequence metrics
- `generated_category_dist.png`: Category distribution chart
- `match_rates.png`: Match rate distributions
- `diversity_metrics.png`: Diversity analysis charts

### 4. `evaluate_generative_model.py`
Evaluates generated sequences against real data across multiple dimensions:
- Life span distributions
- Number of children
- Income distributions
- Gender distributions
- Birth patterns (babies within 3 years for couples)
- Employment patterns (number of jobs)

**Usage:**
```bash
python evaluate_generative_model.py \
    --generated_h5 /path/to/generated_sequences.h5 \
    --real_h5 /projects/0/prjs1589/stonybrook/fake_data_v0/step5/encoding=nomlm/encoded.h5 \
    --vocab_path /projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv \
    --output_dir /projects/0/prjs1589/stonybrook/llm/gen_out/evaluation/ \
    --max_sequences 1000
```

**Outputs:**
- `evaluation_summary.json`: Comparison statistics
- `lifespan_distribution.png`: Life span comparison
- `children_distribution.png`: Children count comparison
- `income_distribution.png`: Income level comparison
- `num_jobs_distribution.png`: Employment comparison

## Workflow

### For Existing Output Files

1. **Repair the file:**
   ```bash
   python repair_generative_output.py \
       /projects/0/prjs1589/stonybrook/llm/gen_out/pretty_tokens_20251212.txt \
       /projects/0/prjs1589/stonybrook/llm/gen_out/pretty_tokens_20251212_repaired.txt
   ```

2. **Analyze the repaired file:**
   ```bash
   python analyze_generative_output.py \
       /projects/0/prjs1589/stonybrook/llm/gen_out/pretty_tokens_20251212_repaired.txt \
       --output_dir /projects/0/prjs1589/stonybrook/llm/gen_out/analysis/
   ```

### For New Generations

Now that `utils.py` is fixed, new generations will automatically produce correctly formatted output (one line per sequence). You can skip the repair step and go straight to analysis.

```bash
# Run generation (this will now produce correct format)
sbatch pop2vec/llm/slurm_scripts/snellius/generative_infer_small.sh

# Analyze the output
python analyze_generative_output.py \
    /projects/0/prjs1589/stonybrook/llm/gen_out/pretty_tokens_NEW.txt \
    --output_dir /projects/0/prjs1589/stonybrook/llm/gen_out/analysis_NEW/
```

### For Model Evaluation

To evaluate your generative model:

```bash
# First, generate 1000 lives from scratch
# Update your hparams file:
#   - num_sequences: 1000
#   - prefix_len: 0  (or very small, like 5)
#   - horizon: 200 (or larger to generate full lives)

# Then evaluate
python evaluate_generative_model.py \
    --generated_h5 /path/to/generated_sequences.h5 \
    --real_h5 /projects/0/prjs1589/stonybrook/fake_data_v0/step5/encoding=nomlm/encoded.h5 \
    --vocab_path /projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv \
    --output_dir /projects/0/prjs1589/stonybrook/llm/gen_out/evaluation/ \
    --max_sequences 1000
```

## Requirements

All scripts require:
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- h5py (for `evaluate_generative_model.py`)

Install with:
```bash
pip install pandas numpy matplotlib seaborn h5py
```

## Professor's Requirements

### ✅ Generate 1000 lives from scratch
Set in hparams:
```yaml
num_sequences: 1000
prefix_len: 5  # Very short prefix
horizon: 500   # Long enough for full life
```

### ✅ Validate distributions
Use `evaluate_generative_model.py` to get:
- Distribution of life spans
- Distribution of number of children
- Distribution of income

### ✅ For couples, check for babies in next 3 years
The evaluator includes `check_birth_in_window()` function that looks for birth events within a 3-year window.

## Notes

1. **Token Encoding:** The evaluation scripts make assumptions about how your data is encoded. You may need to adjust the extraction functions in `evaluate_generative_model.py` based on your specific vocabulary and encoding scheme.

2. **Category Detection:** The scripts look for specific patterns in token names and categories. Review the `LifeSequenceEvaluator` class and adjust patterns as needed.

3. **Performance:** For large files, the analysis scripts can take several minutes. Consider using `--max_sequences` to limit processing during testing.

## Troubleshooting

**Q: The repair script isn't working on my file**
A: Check that your file follows the expected format: `TYPE (Sequence N),count,token1,token2,...`

**Q: The analysis shows very low match rates**
A: This is expected for generative models! Match rates measure exact token matches with the original continuation, which is very strict. Focus on category match rates and distribution similarities instead.

**Q: The evaluation script shows errors about missing categories**
A: You may need to adjust the extraction functions in `LifeSequenceEvaluator` to match your specific vocabulary encoding.

## Future Improvements

- Add more sophisticated metrics (perplexity, BLEU-like scores)
- Support for temporal alignment in sequence comparison
- Interactive visualization dashboards
- Automated report generation
