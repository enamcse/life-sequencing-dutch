# Complete Workflow Guide for Generative Evaluation Pipeline

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Step-by-Step Workflow](#step-by-step-workflow)
4. [Key Concepts](#key-concepts)
5. [Configuration Reference](#configuration-reference)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The generative evaluation pipeline runs experiments that:
1. **Generate sequences** using trained models on specific datasets
2. **Compute statistics** comparing generated vs original sequences
3. **Export results** to CSV for analysis

The **Supervisor v2** automates job submission, monitoring, and result aggregation on the SLURM cluster.

---

## File Structure

```
gen_eval/
├── config/
│   ├── registry.yaml                 # ⭐ MUST FILL: Dataset/model paths
│   └── generative_datasets_config.yaml  # Dataset creation config
├── scripts/
│   ├── supervisor_v2.py              # Main supervisor script
│   ├── supervisor_v2_config.yaml     # ⭐ EDIT: Models, datasets, GPUs, priorities
│   ├── supervisor_v2.sh              # SLURM wrapper to submit supervisor
│   └── create_generative_datasets.py # Creates GD0-GD4, GDB0-GDB4 datasets
├── supervisor_state/                 # Created at runtime
│   ├── pipeline_state_v2.json        # Job state (persists across restarts)
│   ├── dashboard_v2.txt              # Human-readable status
│   └── gpu_config.yaml               # Runtime GPU config (editable live)
└── src/
    ├── generate_sequences.py
    └── compute_statistics.py
```

---

## Step-by-Step Workflow

### Phase 1: Prepare Datasets (One-Time Setup)

#### Step 1.1: Edit Dataset Creation Config

```bash
vim gen_eval/config/generative_datasets_config.yaml
```

Verify:
- [ ] `source_h5.primary` points to your main H5 file
- [ ] `source_h5.birthday` points to your birthday-token H5 file (optional)
- [ ] `output_dir` is where you want datasets saved
- [ ] `vocab_path` points to your vocabulary CSV

#### Step 1.2: Create Datasets

```bash
# Dry run first (see what would be created)
python create_generative_datasets.py --config ../config/generative_datasets_config.yaml --dry-run

# Create all datasets
python create_generative_datasets.py --config ../config/generative_datasets_config.yaml

# Or create specific ones
python create_generative_datasets.py --config ../config/generative_datasets_config.yaml --datasets GD0 GD1
```

This creates:
- `GD0/GD0.h5` - Childhood/Young (age 1-30)
- `GD1/GD1.h5` - Middle-age (30-49)
- `GD2/GD2.h5` - Late middle-age (50-69)
- `GD3/GD3.h5` - Old age (70-99)
- `GD4/GD4.h5` - Mixed sampling
- `GDB0/GDB0.h5` through `GDB4/GDB4.h5` - Birthday-token versions

**NOTE:** If you have old files named `GD0B` instead of `GDB0`, rename them:
```bash
mv GD0B GDB0
mv GD1B GDB1
# etc.
```

---

### Phase 2: Configure the Registry (Critical!)

#### Step 2.1: Fill in Dataset Paths

```bash
vim gen_eval/config/registry.yaml
```

Replace all `null` values with actual paths:

```yaml
datasets:
  GD0: /projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/datasets/GD0/GD0.h5
  GD1: /projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/datasets/GD1/GD1.h5
  GD2: /projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/datasets/GD2/GD2.h5
  GD3: /projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/datasets/GD3/GD3.h5
  GD4: /projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/datasets/GD4/GD4.h5
  GDB0: /projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/datasets/GDB0/GDB0.h5
  GDB1: /projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/datasets/GDB1/GDB1.h5
  # ... etc
```

#### Step 2.2: Fill in Model Paths

```yaml
models:
  Gen-medium:
    checkpoint: /path/to/gen_medium.ckpt    # MUST FILL
    vocab: /projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv
    uses_birthday: false
    
  Gen-BASE:
    checkpoint: /path/to/gen_base.ckpt      # MUST FILL
    vocab: /projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv
    uses_birthday: false
    
  Gen-BASE-bd:
    checkpoint: /path/to/gen_base_bd.ckpt   # MUST FILL
    vocab: /projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv
    uses_birthday: true
```

#### Step 2.3: Verify Model-Dataset Compatibility

The registry defines which datasets each model should use:

```yaml
model_dataset_compatibility:
  Gen-medium: [GD0, GD1, GD2, GD3, GD4]       # Non-birthday model → GD*
  Gen-BASE: [GD0, GD1, GD2, GD3, GD4]
  Gen-medium-bd: [GDB0, GDB1, GDB2, GDB3, GDB4]  # Birthday model → GDB*
  Gen-BASE-bd: [GDB0, GDB1, GDB2, GDB3, GDB4]
```

---

### Phase 3: Configure the Supervisor

#### Step 3.1: Edit Supervisor Config

```bash
vim gen_eval/scripts/supervisor_v2_config.yaml
```

**Models to evaluate:**
```yaml
models:
  - Gen-medium
  - Gen-BASE
```

**Datasets to use:**
```yaml
datasets:
  - GD0
  - GD1
  - GD2
  - GD3
  - GD4
```

**GPU configuration:**
```yaml
gpus:
  ossc9424vm1:
    - 0
    - 1
    - 2
    - 3
```

**Priority configuration (critical!):**
```yaml
priority:
  n: 100                    # num_people
  c: 100                    # num_generations
  max_priority_level: 1     # Start with 1, increase later
  min_priority: 990         # Don't change unless you know what you're doing
```

---

### Phase 4: Run the Supervisor

#### Step 4.1: Submit Supervisor Job

```bash
cd gen_eval/scripts/
sbatch supervisor_v2.sh
```

#### Step 4.2: Monitor Progress

```bash
# Watch the dashboard
watch -n 10 cat ../supervisor_state/dashboard_v2.txt

# Check logs
tail -f /projects/0/prjs1589/stonybrook/logs/supervisor_v2-*.out

# Check SLURM queue
squeue -u $USER
```

#### Step 4.3: Add More Experiments (Incremental Priority)

Once priority level 1 completes:

1. Edit config:
   ```yaml
   priority:
     max_priority_level: 2  # Was 1, now 2
   ```

2. Restart supervisor:
   ```bash
   # The old one may still be running
   scancel <old_job_id>
   sbatch supervisor_v2.sh
   ```

3. **Previous results are preserved!** The state file tracks completed jobs.

#### Step 4.4: Modify GPUs at Runtime

While supervisor is running, edit:
```bash
vim ../supervisor_state/gpu_config.yaml
```

```yaml
gpus:
  ossc9424vm1: [0, 1, 2, 3]
  ossc9424vm2: [0, 1]  # Add new node
```

Changes are detected automatically within 60 seconds.

---

## Key Concepts

### Priority System Explained

The priority system controls which experiments run and in what order.

**Priority Levels (1-10):**

| Level | Parameter Added | Total Experiments (5 models × 5 datasets) |
|-------|----------------|-------------------------------------------|
| 1 | Base: t=0.8, k=20, h=20, g=100 | 25 |
| 2 | +k=1 | 50 |
| 3 | +k=v (full vocab) | 75 |
| 4 | +t=0.1 | 150 |
| 5 | +t=1.0 | 225 |
| 6 | +k=10 | 300 |
| 7 | +t=0.3 | 400 |
| 8 | +k=5 | 500 |
| 9 | +h=10 | 1000 |
| 10 | +g=50 | 2000 |

**`max_priority_level`** controls how many priority levels to include.
- `max_priority_level: 1` → Only base experiments (25)
- `max_priority_level: 5` → Levels 1-5 (225 experiments)

**`min_priority`** is the internal priority score filter (advanced).
- Each experiment gets a priority score: `1000 - level`
- Level 1 experiments have priority 999
- Level 10 experiments have priority 990
- `min_priority: 990` includes all levels

**Recommendation:** Just use `max_priority_level` and leave `min_priority: 990`.

### State Persistence

The supervisor saves state to `pipeline_state_v2.json`:
- Tracks all experiments and their status
- Persists across restarts
- **Completed jobs are NOT re-run**

To force a fresh start (loses all progress):
```bash
rm ../supervisor_state/pipeline_state_v2.json
```

### Path Resolution Order

Paths are resolved in this order (first non-null wins):
1. Command-line argument
2. Config file (`supervisor_v2_config.yaml`)
3. Default value (hardcoded)

Example config file paths:
```yaml
# Uncomment to override defaults:
slurm_dir: /custom/path/slurm_scripts
output_dir: /custom/path/output
```

---

## Configuration Reference

### supervisor_v2_config.yaml

| Key | Type | Description |
|-----|------|-------------|
| `models` | list | Model names to evaluate |
| `datasets` | list | Dataset names to use |
| `gpus` | dict | Node → GPU indices mapping |
| `priority.n` | int | num_people for all experiments |
| `priority.c` | int | num_generations for all experiments |
| `priority.max_priority_level` | int | Include levels 1 to this value |
| `priority.min_priority` | int | Skip experiments below this score |
| `poll_interval_seconds` | int | How often to check job status |
| `auto_submit_stats` | bool | Auto-submit stats after generation |
| `auto_resubmit_failed` | bool | Auto-retry failed jobs |
| `max_retries` | int | Max retry attempts |
| `auto_aggregate` | bool | Auto-create result CSVs |

### registry.yaml

| Section | Description |
|---------|-------------|
| `datasets` | Dataset name → H5 file path mapping |
| `models` | Model name → {checkpoint, vocab, uses_birthday} |
| `model_dataset_compatibility` | Model name → list of compatible datasets |
| `default_vocab` | Fallback vocab path |

---

## Troubleshooting

### "Dataset 'GD0' not found or null in registry"

**Cause:** `registry.yaml` has `null` for the dataset path.
**Fix:** Fill in the actual path:
```yaml
datasets:
  GD0: /actual/path/to/GD0/GD0.h5
```

### "Job disappeared from queue without completion markers"

**Cause:** Job crashed or was cancelled without writing completion message.
**Fix:** Check the job's log file for errors:
```bash
cat /projects/0/prjs1589/stonybrook/logs/*GD0*.err
```

### Previous results being overwritten

**Cause:** State file was deleted.
**Fix:** The state file should persist. If you need to reset specific experiments:
1. Edit `pipeline_state_v2.json`
2. Change status from `completed` to `not_started`

### GPUs not being used

**Cause:** `gpu_config.yaml` doesn't match available hardware.
**Fix:** Check that nodes/GPUs in config actually exist:
```bash
sinfo -p gpu_h100 -N -l
```

### Config file path changes not taking effect

**Cause:** You edited `supervisor_v2_config.yaml` but paths aren't applied.
**Fix:** 
1. Make sure paths are not commented out
2. Restart the supervisor (it reads config at startup)

---

## Quick Reference Commands

```bash
# Submit supervisor
sbatch supervisor_v2.sh

# Watch progress
watch -n 10 cat ../supervisor_state/dashboard_v2.txt

# Check queue
squeue -u $USER

# Stop supervisor
scancel <job_id>

# Reset state (WARNING: loses progress)
rm ../supervisor_state/pipeline_state_v2.json

# Check results
ls /projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/exports/
```
