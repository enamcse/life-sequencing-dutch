# H5 Sequence Statistics & Extraction Tools

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Data Format](#data-format)
4. [Tools Description](#tools-description)
5. [Lifespan Criteria System](#lifespan-criteria-system)
6. [Paired File Extraction](#paired-file-extraction)
7. [Usage Examples](#usage-examples)
8. [Command Line Reference](#command-line-reference)
9. [SLURM Integration](#slurm-integration)
10. [Config File Format](#config-file-format)
11. [Performance Optimization](#performance-optimization)
12. [Workflow Examples](#workflow-examples)

---

## Overview

This toolkit provides high-performance tools for analyzing and extracting sequences from large HDF5 datasets used in life-sequence modeling. The tools are designed to:

- **Analyze** sequence properties across millions of records
- **Filter** sequences based on age at specific positions
- **Extract** subsets matching complex multi-position criteria
- **Support** paired file extraction (original + birthday token versions)
- **Verify** sequence_id matching between paired files

### Key Features

- ⚡ **Vectorized Operations**: NumPy-based processing, no Python loops over sequences
- 🔄 **Parallel Processing**: Multi-process chunk processing for large datasets
- 📊 **Comprehensive Statistics**: Age distributions, decade breakdowns, pair frequencies
- 🎯 **Flexible Filtering**: Position-based age criteria via JSON/YAML configs
- 👯 **Paired Extraction**: Extract from two aligned H5 files simultaneously
- ✅ **ID Verification**: Verify sequence_ids match between paired files
- 📝 **Config Generation**: Statistics script generates configs for extraction
- 🖥️ **SLURM Ready**: Pre-configured job scripts for HPC clusters

---

## Core Concepts

### What is a "Lifespan Sequence"?

In life-sequence modeling, each person's life events are encoded as a sequence of tokens. Each token has an associated age (when the event occurred). A typical sequence has 1024 positions:

```
Position:  0    1    2    3    4    5    6    7    ...  1023
Token:     [CLS][BOS][SEP][PAD][PAD][PAD][EVT][EVT]...  [EVT]
Age:       -1   -1   -1   0    0    0    0    1   ...   95
```

- **Position 0-5**: Special tokens (CLS, BOS, SEP, padding)
- **Position 6**: First real life event token (typically childhood, age 0-9)
- **Position 7-1023**: Life events progressing through age

### Position-Based Age Criteria

The core filtering concept: at each position in the sequence, we expect the age to fall within a specific range. For a "full lifespan" sequence:

| Position | Expected Age Range | Life Stage |
|----------|-------------------|------------|
| 6        | 0-9               | Childhood  |
| 100      | 10-19             | Teens      |
| 200      | 20-29             | Twenties   |
| 300      | 30-39             | Thirties   |
| 400      | 40-49             | Forties    |
| 500      | 50-59             | Fifties    |
| 600      | 60-69             | Sixties    |
| 700      | 70-79             | Seventies  |
| 800      | 80-89             | Eighties   |
| 900      | 90-99             | Nineties   |
| 1000     | 90-99             | End of life|

This allows us to find sequences that represent complete life trajectories from childhood to old age.

---

## Data Format

### HDF5 Structure

Both tools expect HDF5 files with an `input_ids` dataset:

```
input_ids: shape (N, 4, 1024)
├── input_ids[:, 0, :] = Token IDs (vocabulary indices)
├── input_ids[:, 1, :] = Absolute positions
├── input_ids[:, 2, :] = Ages (years)
└── input_ids[:, 3, :] = Segment IDs

sequence_id: shape (N,)  # Optional, used for paired file verification
└── RINPERSOON IDs for each sequence
```

Where:
- `N` = Number of sequences (e.g., 27 million people)
- `4` = Four parallel channels of information
- `1024` = Sequence length (positions 0-1023)

---

## Tools Description

### 1. h5_sequence_statistics.py

**Purpose**: Analyze sequence properties and generate statistics reports.

**Capabilities**:
- Count sequences matching various criteria
- Compute age distributions at key positions
- Find decade breakdowns across the dataset
- Scan position ranges to understand sequence structure
- Generate config files for the extractor

### 2. h5_sequence_extractor.py

**Purpose**: Extract sequences matching specific criteria to new HDF5 files.

**Capabilities**:
- Filter by legacy named criteria (childhood_start, full_length, end_of_life)
- Filter by config-based position-age criteria
- **Paired file extraction** - extract same indices from two aligned H5 files
- **Sequence ID verification** - verify RINPERSOON IDs match between files
- Random sample from matching sequences
- Efficient batch extraction with sorted indices

---

## Lifespan Criteria System

### Default Lifespan Criteria

The tools include a built-in "lifespan" criteria set representing decade progression through life:

```json
{
  "name": "lifespan_decade_progression",
  "description": "Sequences with childhood start and decade progression through life",
  "position_age_criteria": [
    {"position": 6, "age_min": 0, "age_max": 9, "label": "childhood"},
    {"position": 100, "age_min": 10, "age_max": 19, "label": "teens"},
    {"position": 200, "age_min": 20, "age_max": 29, "label": "twenties"},
    {"position": 300, "age_min": 30, "age_max": 39, "label": "thirties"},
    {"position": 400, "age_min": 40, "age_max": 49, "label": "forties"},
    {"position": 500, "age_min": 50, "age_max": 59, "label": "fifties"},
    {"position": 600, "age_min": 60, "age_max": 69, "label": "sixties"},
    {"position": 700, "age_min": 70, "age_max": 79, "label": "seventies"},
    {"position": 800, "age_min": 80, "age_max": 89, "label": "eighties"},
    {"position": 900, "age_min": 90, "age_max": 99, "label": "nineties"},
    {"position": 1000, "age_min": 90, "age_max": 99, "label": "nineties_end"}
  ]
}
```

### Legacy Criteria (Backward Compatibility)

For simpler use cases, the extractor supports named criteria:

| Criterion | Description |
|-----------|-------------|
| `childhood_start` | Age at position 6 is 0-9 |
| `full_length` | Token or age at position 1023 is non-zero |
| `end_of_life` | Age at position 1023 is 70-99 |
| `decade_70` | Age at position 1023 is 70-79 |
| `decade_80` | Age at position 1023 is 80-89 |
| `decade_90` | Age at position 1023 is 90-99 |
| `all` | Shorthand for childhood_start + full_length + end_of_life |

---

## Paired File Extraction

### Concept

We maintain two versions of H5 sequence files:
1. **Original sequences** - The standard encoded life events
2. **Birthday token sequences** - Same sequences with birthday tokens inserted

Both files have:
- Same number of entries (N sequences)
- Same index ordering
- `sequence_id` dataset containing RINPERSOON IDs for verification

### How Paired Extraction Works

1. **Criteria applied on primary file only** - The filtering logic runs on the original/primary H5 file
2. **Same indices extracted from both files** - Once matching indices are found, they are used to extract from both files
3. **Sequence ID verification** - Before extraction, the tool verifies that `sequence_id` values match at each selected index
4. **Two output files generated** - One for each input file, with matching sequences

### Verification Process

```
Primary file:     encoded.h5
Secondary file:   encoded_birthday.h5

For each selected index i:
  primary_id   = primary_file['sequence_id'][i]
  secondary_id = secondary_file['sequence_id'][i]
  
  if primary_id != secondary_id:
    → Report mismatch
    → Save to *_mismatches.json
    → Abort extraction
```

### Output Files (Paired Mode)

```
extracted_original_20260106_120000.h5         # Primary output
extracted_birthday_20260106_120000.h5         # Secondary output
extracted_original_20260106_120000_paired_summary.txt
extracted_original_20260106_120000_criteria.json
```

---

## Usage Examples

### Statistics Script

#### Basic Statistics
```bash
python h5_sequence_statistics.py --h5_file encoded.h5 --output stats.txt
```

#### With Lifespan Criteria Check
```bash
python h5_sequence_statistics.py \
    --h5_file encoded.h5 \
    --output stats.txt \
    --lifespan_check \
    --generate_config lifespan_criteria.json
```

### Extractor Script - Single File

#### Legacy Mode - All Three Criteria
```bash
python h5_sequence_extractor.py \
    --h5_file encoded.h5 \
    --output extracted.h5 \
    --n_sequences 10000 \
    --criteria childhood_start,full_length,end_of_life
```

#### Config Mode - Position-Based Criteria
```bash
python h5_sequence_extractor.py \
    --h5_file encoded.h5 \
    --output extracted.h5 \
    --n_sequences 10000 \
    --config lifespan_criteria.json
```

### Extractor Script - Paired Files

#### Extract from Both Original and Birthday Token Files
```bash
python h5_sequence_extractor.py \
    --h5_file encoded.h5 \
    --output extracted_original.h5 \
    --h5_file_secondary encoded_birthday.h5 \
    --output_secondary extracted_birthday.h5 \
    --n_sequences 10000 \
    --config lifespan_criteria.json
```

#### Skip ID Verification (if you're sure indices match)
```bash
python h5_sequence_extractor.py \
    --h5_file encoded.h5 \
    --output extracted_original.h5 \
    --h5_file_secondary encoded_birthday.h5 \
    --output_secondary extracted_birthday.h5 \
    --n_sequences 10000 \
    --config lifespan_criteria.json \
    --skip_id_verification
```

#### Generate Default Config Only
```bash
python h5_sequence_extractor.py --generate_config lifespan_criteria.json
```

---

## Command Line Reference

### h5_sequence_statistics.py

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--h5_file` | str | Required | Path to input HDF5 file |
| `--output` | str | Auto | Output file for statistics report |
| `--n_workers` | int | 8 | Number of parallel workers |
| `--chunk_size` | int | 100000 | Sequences per processing chunk |
| `--sequential` | flag | False | Use sequential processing (memory-safe) |
| `--end_pos` | int | 1023 | End position to check |
| `--scan_range` | str | None | Range to scan, e.g., "1000-1023" |
| `--find_real_end` | flag | False | Find actual last non-zero position |
| `--lifespan_check` | flag | False | Check default lifespan criteria |
| `--config` | str | None | Path to custom criteria config |
| `--generate_config` | str | None | Generate config file for extractor |

### h5_sequence_extractor.py

#### Primary File Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--h5_file` | str | Required* | Path to primary HDF5 file (criteria applied here) |
| `--output` | str | Required* | Path to primary output HDF5 file |

#### Secondary File Arguments (Paired Mode)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--h5_file_secondary` | str | None | Path to secondary HDF5 file (e.g., birthday token version) |
| `--output_secondary` | str | None | Path to secondary output HDF5 file |
| `--skip_id_verification` | flag | False | Skip sequence_id verification between files |

#### Extraction Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n_sequences` | int | 10000 | Number of sequences to extract |
| `--criteria` | str | None | Comma-separated legacy criteria |
| `--config` | str | None | Path to JSON/YAML config file |
| `--generate_config` | str | None | Generate default config and exit |

#### Performance Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n_workers` | int | 8 | Number of parallel workers |
| `--chunk_size` | int | 100000 | Sequences per processing chunk |
| `--sequential` | flag | False | Use sequential processing |
| `--seed` | int | 42 | Random seed for sampling |

#### Output Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--find_only` | flag | False | Only find, don't extract |
| `--save_indices` | str | None | Save matching indices to file |

*Not required when using `--generate_config`

---

## SLURM Integration

### SLURM Script Structure

The SLURM script (`run_h5_extractor.sh`) has a clear parameter section at the top:

```bash
# ============================================================================
# PARAMETERS - EDIT THIS SECTION
# ============================================================================

# ----------------------------------------------------------------------------
# REQUIRED PARAMETERS
# ----------------------------------------------------------------------------
H5_FILE_PRIMARY="/path/to/encoded.h5"
H5_FILE_SECONDARY="/path/to/encoded_birthday.h5"  # Set to "" for single file
OUTPUT_DIR="/path/to/output"
N_SEQUENCES=10000

# ----------------------------------------------------------------------------
# OPTIONAL PARAMETERS
# ----------------------------------------------------------------------------
CRITERIA_MODE="config"      # "config" or "legacy"
CONFIG_FILE=""              # Set to "" for default lifespan criteria
LEGACY_CRITERIA="childhood_start,full_length,end_of_life"
N_WORKERS=8
CHUNK_SIZE=100000
SEQUENTIAL="false"
SEED=42
SKIP_ID_VERIFICATION="false"
OUTPUT_PREFIX="extracted"
```

### Usage Modes

#### 1. Edit Parameters and Run
```bash
# Edit the PARAMETERS section in the script
vim run_h5_extractor.sh

# Run
sbatch run_h5_extractor.sh
```

#### 2. Generate Config File
```bash
sbatch run_h5_extractor.sh generate_config
# or specify output path:
sbatch run_h5_extractor.sh generate_config /path/to/criteria.json
```

#### 3. Command Line Overrides
```bash
sbatch run_h5_extractor.sh /path/to/encoded.h5 /path/to/output.h5 10000
```

### Output Files (Paired Mode)

```
extracted_original_YYYYMMDD_HHMMSS.h5
extracted_birthday_YYYYMMDD_HHMMSS.h5
extracted_original_YYYYMMDD_HHMMSS_paired_summary.txt
extracted_original_YYYYMMDD_HHMMSS_criteria.json
```

---

## Config File Format

### JSON Config Structure

```json
{
  "name": "my_criteria",
  "description": "Description of what these criteria select",
  "position_age_criteria": [
    {
      "position": 6,
      "age_min": 0,
      "age_max": 9,
      "label": "childhood"
    },
    {
      "position": 500,
      "age_min": 50,
      "age_max": 59,
      "label": "fifties"
    }
  ],
  "statistics": {
    "total_sequences": 10000000,
    "matching_all_criteria": 150000,
    "match_percentage": "1.5000%"
  }
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Human-readable name |
| `description` | No | What the criteria select |
| `position_age_criteria` | **Yes** | Array of criterion objects |
| `statistics` | No | Added by statistics script |

### Criterion Object

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `position` | **Yes** | int | Sequence position (0-1023) |
| `age_min` | **Yes** | int | Minimum age (inclusive) |
| `age_max` | **Yes** | int | Maximum age (inclusive) |
| `label` | No | str | Human-readable label |

---

## Performance Optimization

### Memory Management

| Parameter | Memory Impact | Recommendation |
|-----------|---------------|----------------|
| `--n_workers` | Higher = more memory | Start with 8, reduce if OOM |
| `--chunk_size` | Higher = more memory | 100K default, reduce to 50K if needed |
| `--sequential` | Lowest memory | Use if parallel fails |

### Speed Tips

1. **Use more workers** on high-memory nodes:
   ```bash
   N_WORKERS=32
   ```

2. **Increase chunk size** if memory allows:
   ```bash
   CHUNK_SIZE=500000
   ```

3. **Avoid sequential mode** unless necessary (much slower)

### Typical Performance

On a 32-core, 256GB node with 27M sequences:
- Statistics: ~10-20 minutes
- Extraction (10K sequences): ~5-10 minutes
- Paired extraction (10K sequences): ~10-15 minutes

---

## Workflow Examples

### Workflow 1: Paired Extraction with Default Lifespan Criteria

```bash
# Step 1: Edit the SLURM script parameters
vim run_h5_extractor.sh

# Set these values:
# H5_FILE_PRIMARY="/data/encoded.h5"
# H5_FILE_SECONDARY="/data/encoded_birthday.h5"
# OUTPUT_DIR="/output"
# N_SEQUENCES=10000
# CRITERIA_MODE="config"
# CONFIG_FILE=""  # Uses default lifespan criteria

# Step 2: Submit the job
sbatch run_h5_extractor.sh

# Step 3: Check output
ls -la /output/
# extracted_original_*.h5
# extracted_birthday_*.h5
# *_paired_summary.txt
```

### Workflow 2: Custom Criteria with Paired Extraction

```bash
# Step 1: Create custom config
cat > my_criteria.json << 'EOF'
{
  "name": "middle_age_focus",
  "description": "Sequences with events in middle age",
  "position_age_criteria": [
    {"position": 6, "age_min": 0, "age_max": 9, "label": "childhood"},
    {"position": 300, "age_min": 30, "age_max": 39, "label": "thirties"},
    {"position": 400, "age_min": 40, "age_max": 49, "label": "forties"},
    {"position": 500, "age_min": 50, "age_max": 59, "label": "fifties"}
  ]
}
EOF

# Step 2: Update SLURM script
# CONFIG_FILE="/path/to/my_criteria.json"

# Step 3: Submit
sbatch run_h5_extractor.sh
```

### Workflow 3: Single File Extraction (No Birthday Version)

```bash
# Set H5_FILE_SECONDARY="" or "None" in the script
# This disables paired mode

vim run_h5_extractor.sh
# H5_FILE_SECONDARY=""

sbatch run_h5_extractor.sh
```

---

## Output Files

### Extractor Script Outputs (Paired Mode)

| File | Description |
|------|-------------|
| `extracted_original_*.h5` | Primary file extracted sequences |
| `extracted_birthday_*.h5` | Secondary file extracted sequences |
| `*_paired_summary.txt` | Paired extraction summary |
| `*_criteria.json` | Copy of criteria used |
| `*_mismatches.json` | Only if ID mismatches found |

### Output H5 File Structure

```
extracted_*.h5
├── input_ids          (N, 4, 1024) - Extracted sequences
├── original_indices   (N,) - Original indices in source file
├── sequence_id        (N,) - RINPERSOON IDs (if available)
└── attrs:
    ├── source_file
    ├── paired_source      (paired mode only)
    ├── file_type          ("primary" or "secondary")
    ├── n_sequences
    ├── seed
    └── extraction_time
```

---

## Troubleshooting

### Sequence ID Mismatch Error

```
ERROR: Found 1234 sequence_id mismatches!
```

**Cause**: The primary and secondary H5 files have different `sequence_id` values at the same indices.

**Solutions**:
1. Verify both files were generated from the same source
2. Check if files were sorted/reordered differently
3. Use `--skip_id_verification` if you're sure the data is aligned

### Out of Memory (OOM) Errors

```bash
# Solution 1: Reduce workers
N_WORKERS=4

# Solution 2: Reduce chunk size
CHUNK_SIZE=50000

# Solution 3: Use sequential mode
SEQUENTIAL="true"
```

### Missing sequence_id Dataset

```
WARNING: Primary file missing 'sequence_id' dataset, skipping verification
```

This is just a warning. The extraction will proceed without ID verification.

---

## License

Part of the pop2vec project for life-sequence modeling research.
