# Mortality Prediction Label Generation

This directory contains scripts for generating mortality prediction labels from background and death registry data.

## Overview

The script generates binary labels for mortality prediction:
- **Label 0**: Person is alive during the observation window
- **Label 1**: Person died during the observation window (Jan 1, 2021 - Dec 31, 2023)

## Key Dates

| Date | Description | daysSinceFirstEvent |
|------|-------------|---------------------|
| Dec 30, 1971 | Genesis Date | 0 |
| Dec 31, 2020 | Cutoff Date | - |
| Jan 1, 2021 | Observation Start | 17534 |
| Dec 31, 2023 | Observation End | 18995 |
| Jul 7, 2023 | Max Available Data | 18817 |

## Label Assignment Logic

For each person in the background file:

1. **Compute age at cutoff date** (Dec 31, 2020)
   - If age is NOT in range [0, 100]: **DISCARD**

2. **Check death registry**:
   - If person has a death record with `daysSinceFirstEvent < 17534` (died before 2021): **DISCARD**
   - If person has a death record with `daysSinceFirstEvent` in `[17534, 18995)` (died 2021-2023): **Label 1**
   - If person has no death record: **Label 0**
   - If person has a death record with `daysSinceFirstEvent >= 18995` (died after 2023): **Label 0** (alive during window)

## Input Files

### Background File (Step 2)
Expected columns:
- `RINPERSOON`: Person identifier
- `year`: Birth year
- `month`: Birth month

Note: Birth day is assumed to be 1st of the month.

### Death File (Step 2)
Expected columns:
- `RINPERSOON`: Person identifier
- `daysSinceFirstEvent`: Days since genesis date (float)
- `age`: Age at death in years (float)

## Output Structure

```
output_dir/
├── all/
│   └── death-after-2020.parquet          # Full dataset
├── subset/
│   └── death-after-2020.parquet          # Random 200k sample
├── all-splits/
│   ├── train/
│   │   └── death-after-2020.parquet      # 70% of full data
│   ├── val/
│   │   └── death-after-2020.parquet      # 10% of full data
│   └── test/
│       └── death-after-2020.parquet      # 20% of full data
├── subset-splits/
│   ├── train/
│   │   └── death-after-2020.parquet      # 70% of subset
│   ├── val/
│   │   └── death-after-2020.parquet      # 10% of subset
│   └── test/
│       └── death-after-2020.parquet      # 20% of subset
└── stats/
    ├── mortality_statistics.csv          # Summary statistics
    ├── generation_stats.csv              # Generation process stats
    └── plots/
        ├── label_distribution.png
        ├── deaths_over_time_monthly.png
        ├── deaths_by_year.png
        ├── age_at_death_distribution.png
        ├── deaths_by_age_group.png
        ├── heatmap_year_vs_age.png
        ├── birth_year_distribution.png
        ├── age_at_cutoff_distribution.png
        ├── mortality_rate_by_age.png
        └── summary_dashboard.png
```

### Output Parquet Schema
- `RINPERSOON`: Person identifier (string/int)
- `is_dead`: Binary label (0 = alive, 1 = dead)

## Usage

### 1. Setup Folder Structure

First, create the required folder structure:

```bash
./setup_folders.sh /projects/0/prjs1589/stonybrook/evaluation/labels/mortality
```

### 2. Run Label Generation

#### Option A: SLURM Submission (Recommended)

Edit the parameters at the top of `run_mortality_labels.sh`, then:

```bash
sbatch run_mortality_labels.sh
```

Or override parameters at submission time:

```bash
sbatch --export=ALL,BACKGROUND_FILE=/path/to/bg.parquet,DEATH_FILE=/path/to/death.parquet run_mortality_labels.sh
```

#### Option B: Direct Python Execution

```bash
python generate_mortality_labels.py \
    --background-file /path/to/background.parquet \
    --death-file /path/to/death.parquet \
    --output-dir /path/to/output \
    --stats-dir /path/to/stats \
    --seed 42 \
    --subset-size 200000
```

### 3. Command Line Options

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--background-file` | Yes | - | Path to background parquet file |
| `--death-file` | Yes | - | Path to death parquet file |
| `--output-dir` | Yes | - | Base output directory |
| `--stats-dir` | Yes | - | Statistics output directory |
| `--seed` | No | 42 | Random seed for reproducibility |
| `--subset-size` | No | 200000 | Maximum subset sample size |
| `--skip-plots` | No | False | Skip generating visualization plots |

## Statistics Output

### mortality_statistics.csv

Contains for each output file:
- `dataset`: File/split name
- `total_count`: Total number of records
- `label_1_count`: Number of deaths (label=1)
- `label_0_count`: Number of alive (label=0)
- `label_1_ratio`: Proportion of deaths
- `label_0_ratio`: Proportion of alive

### Visualization Plots

1. **label_distribution.png**: Pie chart of label distribution
2. **deaths_over_time_monthly.png**: Monthly death counts
3. **deaths_by_year.png**: Annual death counts (2021, 2022, 2023)
4. **age_at_death_distribution.png**: Histogram of age at death
5. **deaths_by_age_group.png**: Deaths by decade (0-9, 10-19, ..., 100+)
6. **heatmap_year_vs_age.png**: Heatmap of deaths by year and age group
7. **birth_year_distribution.png**: Birth year distribution by label
8. **age_at_cutoff_distribution.png**: Age at cutoff date by label
9. **mortality_rate_by_age.png**: 3-year mortality rate by age group
10. **summary_dashboard.png**: Combined summary visualization

## Important Notes

1. **Folder Pre-requisite**: The folders `all`, `subset`, `all-splits`, and `subset-splits` must already exist. The script will NOT create them. This is a safety measure to prevent overwriting existing datasets.

2. **Reproducibility**: All random operations use the specified seed for reproducibility.

3. **Memory Requirements**: For large datasets, ensure sufficient memory (64GB recommended).

4. **Split Ratios**: Train:Val:Test = 70:10:20

## Troubleshooting

### Error: Required folder does not exist
Run `setup_folders.sh` first to create the folder structure.

### Error: Missing required column
Check that your input files have the expected column names (case-sensitive for some columns).

### Memory Error
Increase SLURM memory allocation or use a subset of the data for testing.
