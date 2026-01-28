# Export for Plotting Guide

## Overview

`export_for_plotting.py` consolidates generative evaluation results from multiple experiment folders into compact CSV files for downstream plotting and analysis.

## Quick Start

```bash
python export_for_plotting.py \
    --output-dir /projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval \
    --export-dir /projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/exports
```

**No config file needed!** The script auto-discovers experiments.

---

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--output-dir` | Yes | Root folder containing all `exp_*` experiment folders |
| `--export-dir` | Yes | Where to save the output CSV files (created if missing) |

---

## Input Structure

The script expects this folder structure:

```
output-dir/
├── exp_n100_c100_h20_g100_k20_t08_GenBASE_GD0/
│   ├── statistics_n100_c100_summary.csv        # Blockwise stats
│   ├── statistics_by_age_n100_c100_summary.csv # Decade stats
│   ├── original_sequences.parquet              # For real token counts
│   ├── ages.parquet                            # For decade mapping
│   └── token_counts_by_decade_n100_c100.csv    # Optional: precomputed real counts
├── exp_n100_c100_h20_g100_k20_t08_GenBASEbd_GDB0/
│   └── ...
├── exp_n100_c100_h20_g100_k20_t08_GenFT_GD0/
│   └── ...
└── ...
```

### Required Files Per Experiment

| File | Purpose | Required? |
|------|---------|-----------|
| `statistics_n*_c*_summary.csv` | Blockwise comparison stats | Yes (for blockwise export) |
| `statistics_by_age_n*_c*_summary.csv` | Decade-based stats | Yes (for decade export) |
| `original_sequences.parquet` | Compute real token counts | Optional |
| `ages.parquet` | Map tokens to age decades | Optional |
| `token_counts_by_decade_n*_c*.csv` | Precomputed real counts | Optional (preferred) |

---

## Output Files

The script creates one file per dataset:

### 1. `blockwise_{dataset}.csv`

Contains both **comparison metrics** (12 row types) and **token frequencies** by prefix length.

**Columns:**
- `prefix_len`: Prefix length (7, 100, 200, ..., 1000)
- `row_type`: Metric type (see below)
- `token_id`: Token ID (-1 for comparisons, actual ID for token_frequency)
- `num_{exp}`: Numerator for each experiment
- `den_{exp}`: Denominator for each experiment
- `real_count`: Real token count (only for token_frequency rows)

**Row Types (12 comparisons):**
- `real_vs_gen_pos`, `real_vs_gen_neg`
- `real_vs_buddy_pos`, `real_vs_buddy_neg`
- `gen_vs_buddy_pos`, `gen_vs_buddy_neg`
- `real_vs_gen_neutral`, `real_vs_buddy_neutral`, `gen_vs_buddy_neutral`
- `real_vs_gen_tie`, `real_vs_buddy_tie`, `gen_vs_buddy_tie`
- `token_frequency` (token-level stats)

### 2. `decade_{dataset}.csv`

Token frequencies grouped by age decade.

**Columns:**
- `decade`: Age decade (e.g., "20s", "30s", "40s")
- `token_id`: Token ID
- `num_{exp}`: Numerator for each experiment
- `den_{exp}`: Denominator for each experiment
- `real_count`: Real token count

---

## Dataset Grouping

Experiments are grouped by dataset suffix:

| Folder Name | Parsed Dataset |
|-------------|----------------|
| `exp_..._GenBASE_GD0` | `GD0` |
| `exp_..._GenBASEbd_GDB0` | `GDB0` |
| `exp_..._GenFT_GD1` | `GD1` |
| `exp_..._GenBASEbd_GD0` | `GDB0` ⚠️ (auto-corrected) |

**Birthday Model Correction:** If a model name contains `bd` but the dataset is `GD*`, it's automatically corrected to `GDB*`.

---

## Example Output

After running on a typical setup:

```
Found 24 experiments across 4 datasets
Datasets: ['GD0', 'GD1', 'GDB0', 'GDB1']

=== GD0: 6 experiments ===
  Real counts from exp_n100_c100_h20_g100_k20_t08_GenBASE_GD0
  blockwise_GD0.csv: 15420 rows
  decade_GD0.csv: 8340 rows

=== GDB0: 6 experiments ===
  Real decade counts from token_counts_by_decade
  Real counts from exp_n100_c100_h20_g100_k20_t08_GenBASEbd_GDB0
  blockwise_GDB0.csv: 14200 rows
  decade_GDB0.csv: 7890 rows

=== Done: /path/to/exports ===
```

---

## Notes

### Real Token Counts

- **For blockwise stats:** Computed from `original_sequences.parquet` + `ages.parquet`
- **For decade stats:** Prefers `token_counts_by_decade_n*_c*.csv` if available, otherwise computed

### Birthday Token Issue

For birthday-token experiments (`GDB*`), real token counts may be **zero** if the original sequences were sampled from non-birthday datasets. This is expected for the current experiment setup and doesn't affect generated token statistics.

### Performance

- The script uses `tqdm` for progress bars
- Large experiments may take a few minutes to process
- Memory usage scales with number of unique tokens

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No experiments found" | Check `--output-dir` points to folder with `exp_*` directories |
| Missing blockwise output | Ensure `statistics_n*_c*_summary.csv` exists (not `by_age`) |
| Missing decade output | Ensure `statistics_by_age_n*_c*_summary.csv` exists |
| Zero real counts | Normal for birthday experiments; check `original_sequences.parquet` |

---

## Dependencies

```python
# Required packages
pandas
numpy
pyarrow
tqdm
```

Install with:
```bash
pip install pandas numpy pyarrow tqdm
```
