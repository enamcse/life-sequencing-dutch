# Generative Model Analysis - Summary

## What Was Done

### 1. Fixed the Bug in `utils.py` ✅
**Problem:** The `pretty_print_tokens` function in `/home/ehassan/life-sequencing-dutch/pop2vec/llm/src/new_code/utils.py` was using `max_per_line=20`, causing sequences longer than 20 tokens to be split across multiple lines.

**Solution:** Modified the function to output ONE LINE per sequence, regardless of length:
- Removed `max_per_line` parameter
- Changed output format to single line: `TITLE,COUNT,token1,token2,...`

**Files Changed:**
- `/home/ehassan/life-sequencing-dutch/pop2vec/llm/src/new_code/utils.py` (lines 475-490)

### 2. Created Repair Script ✅
**Script:** `/home/ehassan/life-sequencing-dutch/pop2vec/llm/scripts/repair_generative_output.py`

**What it does:**
- Merges continuation lines from old malformed files
- Removes duplicates from parallel GPU execution
- Ensures proper alternating pattern (ORIGINAL, GENERATED, ORIGINAL, GENERATED)

**Usage:**
```bash
python repair_generative_output.py <input_file> <output_file>
```

### 3. Created Analysis Script ✅
**Script:** `/home/ehassan/life-sequencing-dutch/pop2vec/llm/scripts/analyze_generative_output.py`

**What it computes:**
- Category distribution of generated vs original tokens
- Token match rates (exact matches)
- Category match rates
- Diversity metrics:
  - Unique token ratio
  - Consecutive repeat ratio
  - Most common token frequency
- Per-sequence statistics

**Outputs:**
- `summary.json` - Overall statistics
- `per_sequence_stats.csv` - Detailed metrics per sequence
- `generated_category_dist.png` - Bar chart of category distribution
- `match_rates.png` - Histograms of match rates
- `diversity_metrics.png` - Diversity analysis charts

**Usage:**
```bash
python analyze_generative_output.py <tokens_file> --output_dir <dir>
```

### 4. Created Evaluation Script ✅
**Script:** `/home/ehassan/life-sequencing-dutch/pop2vec/llm/scripts/evaluate_generative_model.py`

**What it evaluates (as per professor's request):**
- ✅ Distribution of life spans
- ✅ Distribution of number of children
- ✅ Distribution of income levels
- ✅ Gender-specific patterns
- ✅ Birth patterns (checking for babies within 3 years for couples)
- ✅ Employment patterns (number of jobs)
- Comparison between generated and real sequences

**Outputs:**
- `evaluation_summary.json` - Comparison statistics
- `lifespan_distribution.png` - Generated vs real
- `children_distribution.png` - Generated vs real
- `income_distribution.png` - Generated vs real
- `num_jobs_distribution.png` - Generated vs real

**Usage:**
```bash
python evaluate_generative_model.py \
    --generated_h5 <path_to_generated.h5> \
    --real_h5 <path_to_real.h5> \
    --vocab_path <vocab.csv> \
    --output_dir <dir> \
    --max_sequences 1000
```

### 5. Created Master Analysis Script ✅
**Script:** `/home/ehassan/life-sequencing-dutch/pop2vec/llm/scripts/run_full_analysis.sh`

Runs repair + analysis in one command:
```bash
./run_full_analysis.sh <input_tokens_file> <output_base_dir>
```

### 6. Created Configuration Files ✅

**For generating 1000 lives:**
`/home/ehassan/life-sequencing-dutch/pop2vec/llm/src/hparams/snellius/generate_1000_lives.txt`
- num_sequences: 1000
- horizon: 500 (full life)
- prefix_len: 5 (minimal)

**For couples expecting babies:**
`/home/ehassan/life-sequencing-dutch/pop2vec/llm/src/hparams/snellius/generate_couples_babies.txt`
- num_sequences: 500
- horizon: 60 (3 years of events)
- prefix_len: 50 (couple's history)

## How to Use

### Quick Start - Analyze Existing Output

```bash
cd /home/ehassan/life-sequencing-dutch/pop2vec/llm/scripts

# Run full analysis on your existing file
./run_full_analysis.sh \
    /projects/0/prjs1589/stonybrook/llm/gen_out/pretty_tokens_20251212.txt \
    /projects/0/prjs1589/stonybrook/llm/gen_out/analysis_20251212
```

### Generate and Evaluate 1000 Lives

```bash
# 1. Generate 1000 lives from scratch
cd ~/life-sequencing-dutch
sbatch pop2vec/llm/slurm_scripts/snellius/generative_infer_small.sh \
    --hparams pop2vec/llm/src/hparams/snellius/generate_1000_lives.txt

# 2. After generation completes, evaluate
python pop2vec/llm/scripts/evaluate_generative_model.py \
    --generated_h5 /path/to/output.h5 \  # You'll need to save H5 format
    --real_h5 /projects/0/prjs1589/stonybrook/fake_data_v0/step5/encoding=nomlm/encoded.h5 \
    --vocab_path /projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv \
    --output_dir /projects/0/prjs1589/stonybrook/llm/gen_out/evaluation_1000_lives/ \
    --max_sequences 1000
```

### Check Couples for Babies

```bash
# 1. Generate next 3 years for couples
sbatch pop2vec/llm/slurm_scripts/snellius/generative_infer_small.sh \
    --hparams pop2vec/llm/src/hparams/snellius/generate_couples_babies.txt

# 2. Analyze the output
cd /home/ehassan/life-sequencing-dutch/pop2vec/llm/scripts
./run_full_analysis.sh \
    /projects/0/prjs1589/stonybrook/llm/gen_out/couples_next_3_years.txt \
    /projects/0/prjs1589/stonybrook/llm/gen_out/couples_analysis

# 3. Check the summary.json for birth-related events
cat /projects/0/prjs1589/stonybrook/llm/gen_out/couples_analysis/analysis/summary.json | grep -A5 "category"
```

## What's New vs What Already Existed

### Already Had:
- `generative_infer.py` - Basic inference script
- `generative_infer_parallel.py` - Parallel version (not fully working)
- `utils.py` with `pretty_print_tokens` (but it had a bug)

### New Scripts Created:
1. ✅ `fix_pretty_print_tokens.py` - Reference for the fix
2. ✅ `repair_generative_output.py` - Repair malformed files
3. ✅ `analyze_generative_output.py` - Comprehensive analysis
4. ✅ `evaluate_generative_model.py` - Model evaluation (professor's requirements)
5. ✅ `run_full_analysis.sh` - Master script
6. ✅ `README_ANALYSIS.md` - Full documentation

### Bug Fixed:
- ✅ `utils.py` - Fixed `pretty_print_tokens` to output one line per sequence

### New Configs:
- ✅ `generate_1000_lives.txt` - For generating 1000 lives
- ✅ `generate_couples_babies.txt` - For couples expecting babies

## Important Notes

### About the Evaluation Script
The `evaluate_generative_model.py` script makes assumptions about how your data is encoded:
- **Life spans:** Looks for year tokens and death markers
- **Children:** Looks for parent-child network layer types
- **Income:** Extracts numerical suffixes from INPATAB tokens

**You may need to adjust these extraction functions** based on your specific vocabulary and encoding scheme. Check the `LifeSequenceEvaluator` class in the script.

### About Match Rates
Don't be alarmed by low token match rates (e.g., 5-10%)! This is **expected and normal** for generative models:
- Token match measures **exact** matches at the same position
- This is extremely strict for sequence generation
- Focus instead on:
  - Category match rates (should be higher)
  - Distribution similarities (from evaluation script)
  - Diversity metrics (avoiding mode collapse)

### Next Steps

1. **Test the repair script** on your existing malformed files
2. **Run analysis** on a few files to understand the metrics
3. **Generate 1000 lives** using the new config
4. **Evaluate the model** using the evaluation script
5. **Adjust extraction functions** in the evaluator based on your data

## Files Location Summary

```
/home/ehassan/life-sequencing-dutch/pop2vec/llm/
├── scripts/
│   ├── fix_pretty_print_tokens.py          (reference)
│   ├── repair_generative_output.py         (repair tool)
│   ├── analyze_generative_output.py        (statistics)
│   ├── evaluate_generative_model.py        (evaluation)
│   ├── run_full_analysis.sh                (master script)
│   └── README_ANALYSIS.md                  (documentation)
├── src/
│   ├── new_code/
│   │   ├── generative_infer.py             (existing, uses fixed utils)
│   │   ├── generative_infer_parallel.py    (existing)
│   │   └── utils.py                        (FIXED - lines 475-490)
│   └── hparams/snellius/
│       ├── generate_1000_lives.txt         (new config)
│       └── generate_couples_babies.txt     (new config)
```

## Questions?

Check the README_ANALYSIS.md file for:
- Detailed usage instructions
- Troubleshooting tips
- Expected output formats
- Extension ideas
