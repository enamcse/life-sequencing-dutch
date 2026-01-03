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
│   ├── generate_sequences.py     # GPU: Generate & save sequences
│   └── compute_statistics.py     # CPU: Compute all statistics
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

# 5. Submit jobs
bash pop2vec/llm/src/new_code/gen_eval/scripts/submit_jobs.sh \
    --experiment exp_n10_c100_h20_g100

# 6. Check progress
python -m pop2vec.llm.src.new_code.gen_eval.scripts.check_progress \
    --experiment exp_n10_c100_h20_g100
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
| g | Prefix gap | 100 (gives 1, 101, 201, ...) |

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

**Key concept: Position-dependent age buckets**

The age decade for each generation is determined by the age at position `prefix_len - 1` (the last token before generation starts). This means:
- The **same person** can contribute to **different decade buckets** at different prefix_lens
- If a person is age 5 at prefix_len 101 and age 15 at prefix_len 501, they contribute to "0-9" bucket for the first block and "10-19" for the second

**Example:**
```
Original sequence:    [CLS] tok1 tok2 tok3 tok4 tok5 ...
Age stream:           0     0    2    3    5    7    ...
Prefix_len=3 → age at position 2 → age=2 → decade "0-9"
Prefix_len=5 → age at position 4 → age=5 → decade "0-9"
Prefix_len=6 → age at position 5 → age=7 → decade "0-9"
```

**Output files:**

1. **`statistics_by_age_n{n}_c{c}_full.csv`** - Statistics grouped by decade
   - Columns: `prefix_len, row_type, token_id, token, 0_9_num, 0_9_den, 10_19_num, 10_19_den, ..., 100plus_num, 100plus_den, total_num, total_den`
   - Instead of per-person columns, has per-decade columns

2. **`statistics_by_age_n{n}_c{c}_summary.csv`** - Summary of by-age statistics
   - Columns: `prefix_len, row_type, token_id, token, total_num, total_den, rate`

This allows analysis of how the model performs across different age groups (0-9, 10-19, 20-29, ..., 90-99, 100+ years old).

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
│  Statistics     │◀────│  Compute Stats    │◀────│   Sequences     │
│    (CSV)        │     │    (CPU)          │     │   (Parquet)     │
└─────────────────┘     └───────────────────┘     └─────────────────┘
```

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
prefix_lengths: [1, 50, 100, 150, 200]
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
