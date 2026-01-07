# Generative Evaluation Pipeline

## Overview

This pipeline evaluates generative models by:
1. Generating token sequences from prefixes (GPU phase)
2. Comparing generated sequences against real futures (CPU phase)
3. Computing comprehensive statistics with per-person detail and summary outputs

## Directory Structure

```
gen_eval/
├── README.md
├── parquet_to_csv.py             # Utility: Convert Parquet to CSV
├── config/
│   ├── models_config.yaml        # Define all models in one file
│   ├── experiments_config.yaml   # Define all experiments in one file
│   ├── events_config.yaml        # Life events config for plotting
│   ├── models/                   # Individual model configurations
│   │   └── model_v1_gen_20251117/
│   │       ├── model.ckpt -> /path/to/checkpoint
│   │       └── config.yaml
│   └── experiments/              # Individual experiment configurations
│       ├── exp_n10_c100_h20_g100.yaml
│       └── exp_n100_c1000_h20_g100.yaml
├── scripts/
│   ├── setup_model.py            # Create model folder structure
│   ├── generate_slurm.py         # Generate SLURM scripts
│   ├── submit_jobs.sh            # Submit SLURM jobs
│   └── check_progress.py         # Monitor job progress
├── src/
│   ├── generate_sequences.py     # Stage 1: Generate & save sequences (GPU)
│   ├── compute_statistics.py     # Stage 2: Compute all statistics (CPU)
│   └── plot_statistics.py        # Stage 3: Generate sanity check plots
└── slurm_scripts/                # Generated SLURM scripts
```

## Quick Start (Config File Method - Recommended)

```bash
cd ~/life-sequencing-dutch

# 1. Edit models config file with your models
vim pop2vec/llm/src/new_code/gen_eval/config/models_config.yaml

# 2. Setup all models at once
python -m pop2vec.llm.src.new_code.gen_eval.scripts.setup_model \
    --config pop2vec/llm/src/new_code/gen_eval/config/models_config.yaml

# 3. Edit experiments config file with your experiments
vim pop2vec/llm/src/new_code/gen_eval/config/experiments_config.yaml

# 4. Generate all SLURM scripts at once
python -m pop2vec.llm.src.new_code.gen_eval.scripts.generate_slurm \
    --config pop2vec/llm/src/new_code/gen_eval/config/experiments_config.yaml

# 4b. (Optional) With GPU assignment (models distributed across GPUs per experiment)
python -m pop2vec.llm.src.new_code.gen_eval.scripts.generate_slurm \
    --config pop2vec/llm/src/new_code/gen_eval/config/experiments_config.yaml \
    --gpus 0,1,2,3

# 4c. (Optional) Multi-node GPU assignment
python -m pop2vec.llm.src.new_code.gen_eval.scripts.generate_slurm \
    --config pop2vec/llm/src/new_code/gen_eval/config/experiments_config.yaml \
    --gpus "ossc9424vm1:0,1;ossc9424vm2:0,1"

# 5. Submit jobs
bash pop2vec/llm/src/new_code/gen_eval/scripts/submit_jobs.sh \
    --experiment exp_n10_c100_h20_g100

# 6. Check progress (table view with timing info)
python -m pop2vec.llm.src.new_code.gen_eval.scripts.check_progress
```

## Alternative: Command Line Method

```bash
# Setup a single model via command line
python -m pop2vec.llm.src.new_code.gen_eval.scripts.setup_model \
    --name model_v1 \
    --checkpoint /path/to/model.ckpt \
    --data /path/to/encoded.h5 \
    --vocab /path/to/vocab.csv

# Generate SLURM scripts with direct parameters
python -m pop2vec.llm.src.new_code.gen_eval.scripts.generate_slurm \
    --models model_v1_gen_20251117 \
    --n 10 --c 100 --h 20 --g 100
```

## Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| n | Number of people | 10, 100, 1000 |
| c | Generations per person | 100, 1000 |
| h | Horizon (tokens to generate) | 20, 50, 100 |
| g | Prefix gap | 100 (gives 7, 100, 200, 300, ...) |

### Prefix Lengths

By default, generation happens at positions: `[7, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]`

- **Position 7**: After demographic tokens `[CLS] (gender, municipality, birth_year, birth_month) [SEP]`
- **Positions 100-1000**: Increment by 100 to cover the full sequence

You can override prefix lengths explicitly in the experiment config:
```yaml
prefix_lengths: [7, 100, 200, 300, 400, 500]  # Custom prefix lengths
```

## Output Format

### Sequences Parquet (Phase 1)

Single file with columns:
- `person_idx`, `rinpersoon_id`: Person identification
- `prefix_len`: Prefix length
- `generation_idx`: Generation index (0 to c-1)
- `original_tokens`: Ground truth continuation
- `generated_tokens`: Model's generated tokens
- `buddy_tokens`: Random buddy's continuation
- `next_tokens`: Next person's continuation

### Ages Parquet (Phase 1)

Single file (`ages.parquet`) with columns:
- `local_idx`: Local index (0 to n-1 for persons, n to 2n-1 for buddies)
- `h5_idx`: Original index in HDF5 file
- `rinpersoon_id`: Person identifier
- `age_stream`: Comma-separated ages for each position in the sequence
- `real_length`: Real length of the sequence (before padding)
- `is_buddy`: Whether this is a buddy sequence

The age at any prefix position `p` can be looked up as `age_stream[p-1]`, allowing
position-dependent decade bucket assignment for by-age statistics.

### Statistics CSV (Phase 2)

**Output files (filenames include n and c for traceability):**

1. **`statistics_n{n}_c{c}_full.csv`** - Full data with per-person columns
   - Columns: `prefix_len, row_type, token_id, token, p0_num, p0_den, p1_num, p1_den, ..., total_num, total_den`
   - 12 comparison rows + V token frequency rows per prefix block

2. **`statistics_n{n}_c{c}_summary.csv`** - Aggregated data only (for easy analysis)
   - Columns: `prefix_len, row_type, token_id, token, total_num, total_den, rate`
   - Same rows as full CSV, but without per-person columns

### By-Age Statistics (Optional)

When `compute_by_age: true` is set, additional statistics files are generated with data grouped by age decade.

**Key concept: Aggregation by life stage, not sequence position**

The by-age statistics aggregate data ACROSS all prefix_lens into decade buckets based on age at each generation point:
- For each generation record, we look up the age at position `prefix_len - 1` (the last token before generation)
- That age determines which decade bucket (0-9, 10-19, ..., 90-99, 100+) the record contributes to
- Statistics are then grouped by decade, giving a view of model performance by life stage

**Why this matters:**
- Regular statistics (by prefix_len) mix different life stages together
- A person's sequence at prefix_len 101 might be age 5, while another person at the same prefix_len might be age 45
- By-age statistics separate these, so we can see how well the model predicts tokens for teenagers vs. middle-aged people
- For example, education-related tokens (CE/CITO scores) should appear more in the 10-19 decade since exams happen around ages 15-18

**Example:**
```
Person A: prefix_len=101, age at position 100 = 5  → contributes to "0-9"
Person A: prefix_len=501, age at position 500 = 25 → contributes to "20-29"
Person B: prefix_len=101, age at position 100 = 45 → contributes to "40-49"
```

**Output files:**

1. **`statistics_by_age_n{n}_c{c}_full.csv`** - Statistics grouped by decade
   - Columns: `decade, row_type, token_id, token, p0_num, p0_den, p1_num, p1_den, ..., total_num, total_den`
   - First column is `decade` (e.g., "0-9", "10-19") instead of `prefix_len`
   - Same per-person columns as regular statistics

2. **`statistics_by_age_n{n}_c{c}_summary.csv`** - Summary by decade
   - Columns: `decade, row_type, token_id, token, total_num, total_den, rate`
   - Same structure as regular summary, but grouped by decade

3. **`token_counts_by_decade_n{n}_c{c}.csv`** - Raw token counts spreadsheet
   - Columns: `decade, N_d, unique_people, token_id, token, simulated_count, real_count`
   - `N_d`: Number of (person, prefix_len) combinations contributing to this decade
   - `unique_people`: Number of distinct people contributing to this decade
   - `simulated_count`: Total count of this token in simulated sequences for this decade
   - `real_count`: Total count of this token in real sequences for this decade
   - Expected totals: `real = N_d × horizon`, `simulated = N_d × horizon × n_generations`
   - Note: Sum of `unique_people` across decades may exceed `n` since a person ages through multiple decades

4. **`decade_summary_n{n}_c{c}.csv`** - Per-decade summary
   - Columns: `decade, N_d, unique_people, total_real_tokens, total_simulated_tokens, expected_real_tokens, expected_simulated_tokens, unique_real_tokens, unique_simulated_tokens`
   - High-level sanity check that token totals match expectations

5. **`age_progression_n{n}_c{c}.csv`** - Age progression by person and prefix_len
   - Columns: `prefix_len, p0, p1, p2, ..., p{n-1}`
   - Each row shows which decade each person falls into at that prefix position
   - Example:
     ```
     prefix_len, p0,   p1,    p2,    ...
     1,          0-9,  0-9,   0-9,   ...
     101,        0-9,  30-39, 10-19, ...
     201,        0-9,  30-39, 20-29, ...
     301,        0-9,  40-49, 20-29, ...
     ```
   - Useful for understanding age distribution across the cohort and verifying age data

This allows analysis of how the model performs across different age groups (0-9, 10-19, 20-29, ..., 90-99, 100+ years old).

### Stage 3: Plotting (Optional)

The plotting stage generates visualization for sanity checking. Plots use **frequency per million tokens** on the y-axis:

```
Real frequency = (real_count / (N_d × horizon)) × 1,000,000
Simulated frequency = (simulated_count / (N_d × horizon × c)) × 1,000,000
```

Where:
- `N_d` = number of (person, prefix_len) combinations in each decade
- `horizon` = tokens per generation window (default 20)
- `c` = number of generations per (person, prefix_len) combination

This gives "how many times token X appeared out of every million tokens generated/observed".

```bash
# Generate plots from token counts
python -m pop2vec.llm.src.new_code.gen_eval.src.plot_statistics \
    --config run_config.yaml

# Or specify files directly
python -m pop2vec.llm.src.new_code.gen_eval.src.plot_statistics \
    --token_counts token_counts_by_decade_n100_c100.csv \
    --events_config config/events_config.yaml \
    --output_dir output/plots \
    --n_generations 100 --horizon 20
```

**Output files:**
- `token_freq_{event}_by_decade.png` - Line plot per life event (real vs simulated)
- `token_freq_{event}_by_decade_log.png` - Log scale version
- `token_freq_all_events_by_decade.png` - All events in one plot
- `real_vs_simulated_scatter.png` - Scatter plot for calibration check

**Events Configuration (`config/events_config.yaml`):**

Life events map to multiple token IDs (e.g., different death-related tokens):

```yaml
life_events:
  death:
    tokens: [1234, 1235, 1236]  # Token IDs for death-related events
    color: "#e41a1c"
    label: "Death"
  retirement:
    tokens: [2001, 2002, 2003]
    color: "#377eb8"
    label: "Retirement"
  school:
    tokens: [3001, 3002]
    color: "#4daf4a"
    label: "School/Education"
```

To auto-generate an events config from vocabulary:
```bash
python -m pop2vec.llm.src.new_code.gen_eval.src.plot_statistics \
    --create_events_config --vocab vocab.csv --output events_config.yaml
```

### Comparison Types (12 rows)

| Row Type | Description |
|----------|-------------|
| `ordered_self_with_pad` | Gen vs Original (ordered, with PAD) |
| `ordered_self_no_pad` | Gen vs Original (ordered, no PAD) |
| `unordered_self_with_pad` | Gen vs Original (unordered, with PAD) |
| `unordered_self_no_pad` | Gen vs Original (unordered, no PAD) |
| `ordered_buddy_with_pad` | Gen vs Buddy (ordered, with PAD) |
| `ordered_buddy_no_pad` | Gen vs Buddy (ordered, no PAD) |
| `unordered_buddy_with_pad` | Gen vs Buddy (unordered, with PAD) |
| `unordered_buddy_no_pad` | Gen vs Buddy (unordered, no PAD) |
| `ordered_next_with_pad` | Gen vs Next person (ordered, with PAD) |
| `ordered_next_no_pad` | Gen vs Next person (ordered, no PAD) |
| `unordered_next_with_pad` | Gen vs Next person (unordered, with PAD) |
| `unordered_next_no_pad` | Gen vs Next person (unordered, no PAD) |

### PAD Token Exclusion Modes

The `no_pad` comparisons support three exclusion modes (configurable via `pad_exclusion_mode`):

| Mode | Description |
|------|-------------|
| `seq1` | Exclude position only if PAD in first sequence (ground truth) |
| `seq2` | Exclude position only if PAD in second sequence (generated) |
| `both` | Exclude position if PAD in either sequence (default) |

### Token Frequency Rows

For each token in the vocabulary:
- `row_type`: `token_frequency`
- `token_id`: Token ID
- `token`: Token name
- `p{i}_num`: Count of this token for person i
- `p{i}_den`: Total tokens generated for person i
- `total_num`: Sum across all people
- `total_den`: Sum of denominators

## Workflow

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  Setup Models   │────▶│ Generate SLURM    │────▶│  Submit Jobs    │
│  (config.yaml)  │     │    Scripts        │     │   (GPU/CPU)     │
└─────────────────┘     └───────────────────┘     └────────┬────────┘
                                                           │
                        ┌───────────────────┐              │
                        │  Check Progress   │◀─────────────┤
                        └───────────────────┘              │
                                                           ▼
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  Plots (Stage 3)│◀────│  Compute Stats    │◀────│   Sequences     │
│    (PNG)        │     │  (CPU - Stage 2)  │     │(GPU - Stage 1)  │
└─────────────────┘     └───────────────────┘     └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  token_freq_*   │     │  statistics_*.csv │     │ sequences.pq    │
│  _by_decade.png │     │  token_counts_*.csv │   │ ages.parquet    │
└─────────────────┘     └───────────────────┘     └─────────────────┘
```

## GPU Assignment

Jobs can be assigned to specific GPU slots (node + GPU index) to optimize resource utilization and prevent memory conflicts.

### Model-wise GPU Assignment

When using the `--gpus` flag, **models within each experiment** are distributed across available GPUs:

```bash
# Simple format: GPUs on default node (ossc9424vm1)
python -m pop2vec.llm.src.new_code.gen_eval.scripts.generate_slurm \
    --config experiments_config.yaml --gpus 0,1,2,3

# Range syntax also works
python -m pop2vec.llm.src.new_code.gen_eval.scripts.generate_slurm \
    --config experiments_config.yaml --gpus 0-3

# Multi-node format: specify GPUs per node
python -m pop2vec.llm.src.new_code.gen_eval.scripts.generate_slurm \
    --config experiments_config.yaml \
    --gpus "ossc9424vm1:0,1;ossc9424vm2:2,3"
```

**How it works:**
- When you submit ONE experiment, different models run on different GPUs
- Models are assigned round-robin across available GPU slots
- Jobs on the same GPU slot run sequentially (via SLURM dependencies)
- Each GPU slot is defined by (node, gpu_index) pair

**Example with 5 models, 4 experiments, and 4 GPUs on ossc9424vm1:**

| model/exp | e1 | e2 | e3 | e4 |
|-----------|----|----|----|----|
| m1 | vm1:GPU0 | vm1:GPU1 | vm1:GPU2 | vm1:GPU3 |
| m2 | vm1:GPU1 | vm1:GPU2 | vm1:GPU3 | vm1:GPU0 |
| m3 | vm1:GPU2 | vm1:GPU3 | vm1:GPU0 | vm1:GPU1 |
| m4 | vm1:GPU3 | vm1:GPU0 | vm1:GPU1 | vm1:GPU2 |
| m5 | vm1:GPU0 | vm1:GPU1 | vm1:GPU2 | vm1:GPU3 |

**When you submit e1:**
- m1 → ossc9424vm1:GPU0
- m2 → ossc9424vm1:GPU1
- m3 → ossc9424vm1:GPU2
- m4 → ossc9424vm1:GPU3
- m5 → ossc9424vm1:GPU0 (waits for m1 to complete)

### Generated SLURM Directives

The generated scripts include proper SLURM directives for node and GPU assignment:

```bash
#SBATCH --nodelist=ossc9424vm1    # Target specific node
...
export CUDA_VISIBLE_DEVICES=0      # Use specific GPU on that node
```

### Sequential GPU Execution

When submitting jobs, the `submit_jobs.sh` script automatically:

1. Reads GPU slot (node:gpu_index) from generated scripts
2. Tracks the last job ID for each GPU slot
3. Adds SLURM dependencies so jobs on the same GPU slot run sequentially
4. Statistics jobs depend on their corresponding generation jobs

This prevents GPU memory conflicts when running large models.

### Config File GPU Specification

You can also specify GPUs in the config file:

```yaml
models:
  - Gen-medium
  - Gen-BASE
  
# Simple format (default node: ossc9424vm1)
gpu_indices: "0,1,2,3"

# Or multi-node format
gpu_indices: "ossc9424vm1:0,1;ossc9424vm2:2,3"

experiments:
  - name: exp_n10_c100_h20_g100
    n: 10
    c: 100
    h: 20
    g: 100
```

## Check Progress

The `check_progress.py` script provides a table view of job status with timing information.

### Table View

```bash
# Show all experiments and models
python -m pop2vec.llm.src.new_code.gen_eval.scripts.check_progress

# Filter by experiment or model
python -m pop2vec.llm.src.new_code.gen_eval.scripts.check_progress \
    -e exp_n10_c100 -m Gen-medium Gen-BASE
```

**Output example:**
```
============================================================
 Generation Jobs
============================================================
Model              exp_n10_c100   exp_n100_c100   exp_n1000_c100
----------------------------------------------------------------
Gen-medium         ✓ 2h15m        ⟳              -
Gen-BASE           ✓ 3h42m        ✓ 4h10m        ⏳
Gen-medium-bd      ✗              -              -
Gen-BASE-bd        ✓ 2h30m        ✓ 3h55m        ⟳

Legend: ✓=completed, ⟳=running, ⏳=pending, ✗=failed, -=not started
```

### Status Symbols

| Symbol | Status | Description |
|--------|--------|-------------|
| ✓ | Completed | Output file exists |
| ⟳ | Running | Job is running in SLURM |
| ⏳ | Pending | Job is queued in SLURM |
| ✗ | Failed | Error log contains errors |
| - | Not started | No job submitted yet |

### Command Options

```bash
# Detailed timing information per job
python -m pop2vec.llm.src.new_code.gen_eval.scripts.check_progress --detailed

# Summary statistics only
python -m pop2vec.llm.src.new_code.gen_eval.scripts.check_progress --summary

# Show only generation or statistics jobs
python -m pop2vec.llm.src.new_code.gen_eval.scripts.check_progress --gen-only
python -m pop2vec.llm.src.new_code.gen_eval.scripts.check_progress --stats-only

# Custom log directory
python -m pop2vec.llm.src.new_code.gen_eval.scripts.check_progress \
    --log-dir /gpfs/ostor/ossc9424/logs
```

### Summary Statistics

The summary includes:
- Total jobs, completed, running, pending, failed, not started counts
- Percentage completion
- Average, min, max duration for completed jobs

## Adding New Models

```bash
# Create model folder structure
python -m pop2vec.llm.src.new_code.gen_eval.scripts.setup_model \
    --name my_new_model \
    --checkpoint /path/to/my_model.ckpt \
    --data /path/to/encoded.h5 \
    --vocab /path/to/vocab.csv
```

This creates:
```
config/models/my_new_model/
├── model.ckpt -> /path/to/my_model.ckpt
└── config.yaml
```

## Creating Custom Experiments

Edit or create a YAML file in `config/experiments/`:

```yaml
experiment_name: "my_experiment"
num_people: 50
num_generations: 500
horizon: 30
# Default prefix_lengths: [7, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# Override with custom values:
prefix_lengths: [7, 100, 200, 300, 400, 500]
top_k: 20
temperature: 1.0
seed: 42

generation:
  partition: "gpu_h100"
  gpus: 1
  cpus: 4
  mem: "64G"
  time: "48:00:00"

statistics:
  partition: "thin"
  cpus: 8
  mem: "32G"
  time: "12:00:00"
```

## H5 Sequence Utilities

Two utility scripts for analyzing and extracting sequences from large HDF5 datasets:

### h5_sequence_statistics.py

Fast, parallel statistics on sequence properties (age distributions, criteria matching, etc.).

```bash
# Basic usage
python -m pop2vec.llm.src.new_code.gen_eval.src.h5_sequence_statistics \
    --h5_file encoded.h5 --output stats_report.txt

# With more workers for faster processing
python -m pop2vec.llm.src.new_code.gen_eval.src.h5_sequence_statistics \
    --h5_file encoded.h5 --n_workers 16 --chunk_size 200000
```

**Output:**
- `*_stats.txt` - Detailed report with:
  - Counts for each criterion (childhood start, full length, end-of-life)
  - All pairwise combinations of criteria
  - Decade distributions at indices 0, 6, and 1023
  - Age distributions at key indices
  - Top (age_0, age_1023) pair frequencies
- `*_age_pairs.csv` - Full (age_0, age_1023) pair data for analysis

**Criteria checked:**
1. **Childhood start**: age at index 6 is in 0-9
2. **Full length**: token at index 1023 is non-zero OR age at index 1023 is non-zero
3. **End-of-life**: age at index 1023 is in 70-99

### h5_sequence_extractor.py

Extract sequences matching specific criteria (e.g., for training subset creation).

```bash
# Extract 10,000 sequences matching all three criteria
python -m pop2vec.llm.src.new_code.gen_eval.src.h5_sequence_extractor \
    --h5_file encoded.h5 --output extracted.h5 --n_sequences 10000

# Custom criteria selection
python -m pop2vec.llm.src.new_code.gen_eval.src.h5_sequence_extractor \
    --h5_file encoded.h5 --output extracted.h5 \
    --criteria childhood_start,full_length,decade_80 \
    --n_sequences 5000

# Just find matching indices without extracting
python -m pop2vec.llm.src.new_code.gen_eval.src.h5_sequence_extractor \
    --h5_file encoded.h5 --output extracted.h5 \
    --find_only --save_indices matching_indices.npy
```

**Available criteria:**
| Criterion | Description |
|-----------|-------------|
| `childhood_start` | Age at index 6 is in 0-9 |
| `full_length` | Token/age at index 1023 is non-zero |
| `end_of_life` | Age at index 1023 is in 70-99 |
| `decade_70` | Age at index 1023 is in 70-79 |
| `decade_80` | Age at index 1023 is in 80-89 |
| `decade_90` | Age at index 1023 is in 90-99 |
| `all` | Shorthand for childhood_start,full_length,end_of_life |

**Output:**
- `extracted.h5` - HDF5 file with selected sequences
  - Same structure as source: `input_ids` with shape (N, 4, 1024)
  - Additional `original_indices` dataset for reference
  - Metadata attributes (source file, seed, etc.)
- `extracted_summary.txt` - Extraction details and statistics

**Example workflow:**

```bash
# 1. First, analyze the full dataset
python -m pop2vec.llm.src.new_code.gen_eval.src.h5_sequence_statistics \
    --h5_file /path/to/encoded.h5 --output analysis.txt --n_workers 16

# 2. Check how many sequences match all criteria
# (see "ALL THREE CRITERIA" in the output report)

# 3. Extract a subset for training/evaluation
python -m pop2vec.llm.src.new_code.gen_eval.src.h5_sequence_extractor \
    --h5_file /path/to/encoded.h5 --output subset_10k.h5 \
    --n_sequences 10000 --seed 42 --n_workers 16
```
