# Supervisor v2 Manual

## Overview

The Pipeline Supervisor v2 is an enhanced automation system for managing generative evaluation experiments on a SLURM cluster. It handles job submission, monitoring, prioritization, and result aggregation with support for runtime configuration updates.

## Key Features

1. **Runtime GPU/Node Modification** - Add or remove GPUs while the supervisor is running
2. **Priority-Based Submission** - Control which experiments run first
3. **Improved Completion Detection** - Customizable markers for job completion
4. **Result Aggregation** - Automatic CSV generation for export
5. **Self-Backup** - Automatic versioning of supervisor scripts

## Quick Start

```bash
# 1. Edit configuration
vim supervisor_v2_config.yaml

# 2. Submit supervisor job
sbatch supervisor_v2.sh

# 3. Monitor progress
cat ../supervisor_state/dashboard_v2.txt

# 4. Modify GPUs at runtime (optional)
vim ../supervisor_state/gpu_config.yaml

# 5. Check exported results
ls /projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/exports/
```

## Configuration

### Main Config: `supervisor_v2_config.yaml`

```yaml
# Models to evaluate
models:
  - Gen-medium
  - Gen-BASE
  - Gen-medium-bd
  - Gen-BASE-bd
  - Gen-BASE-bd-partial

# Datasets (GD0-GD4 for normal, GDB0-GDB4 for birthday-token)
datasets:
  - GD0  # Childhood/Young (age 1-30)
  - GD1  # Middle-age (30-49)
  - GD2  # Late middle-age (50-69)
  - GD3  # Old age (70-99)
  - GD4  # Mixed sampling

# GPU configuration
gpus:
  ossc9424vm1: [0, 1, 2, 3]
  ossc9424vm2: [0, 1, 2, 3]

# Priority configuration
priority:
  n: 100  # num_people
  c: 100  # num_generations
  max_priority_level: 5  # Include experiments up to level 5
  min_priority: 990      # Skip jobs below this priority
```

### Priority Levels

| Level | Parameters Added | Total Experiments |
|-------|-----------------|-------------------|
| 1 | Base: t=0.8, k=20, h=20, g=100 | 25 |
| 2 | +k=1 | 50 |
| 3 | +k=v (vocab size) | 75 |
| 4 | +t=0.1 | 150 |
| 5 | +t=1.0 | 225 |
| 6 | +k=10 | 300 |
| 7 | +t=0.3 | 400 |
| 8 | +k=5 | 500 |
| 9 | +h=10 | 1000 |
| 10 | +g=50 | 2000 |

## Runtime GPU Modification

To add or remove GPUs while the supervisor is running, edit the runtime GPU config file:

```bash
# Edit the runtime GPU config
vim ../supervisor_state/gpu_config.yaml
```

Example `gpu_config.yaml`:

```yaml
# Add a new node
gpus:
  ossc9424vm1:
    - 0
    - 1
    - 2
    - 3
  ossc9424vm2:  # New node added
    - 0
    - 1
```

To remove a GPU from rotation:
- Remove it from the list
- Jobs currently running on that GPU will complete
- No new jobs will be assigned to that GPU

Changes are detected automatically every poll interval (default: 60 seconds).

## Experiment Naming Convention

All experiments follow this naming convention:

```
exp_n{n}_c{c}_h{h}_g{g}_k{k}_t{temp}_{model}_{dataset}
```

Example:
```
exp_n100_c100_h20_g100_k20_t08_GenBASE_GD0
```

Where:
- `n` = num_people (100)
- `c` = num_generations (100)
- `h` = horizon (20)
- `g` = gap/prefix_gap (100)
- `k` = top_k (20, or `v` for vocab size)
- `t` = temperature (0.8, written as "08")
- Model name (special chars removed)
- Dataset name (GD0, GD1, etc.)

### Job Naming

SLURM jobs are named with a sequential number prefix:

```
001.exp_n100_c100_h20_g100_k20_t08_GenBASE_GD0
002.exp_n100_c100_h20_g100_k20_t08_GenBASE_GD1
```

This ensures unique job names in the SLURM queue and logs.

## Completion Detection

The supervisor detects job completion through:

1. **Output Files** - Checks for required output files
2. **Log Markers** - Parses log files for completion messages

### Customizing Completion Markers

To add new completion markers, edit `supervisor_v2.py`:

```python
# Find this section near the top of the file
COMPLETION_MARKERS = {
    'generation': [
        ("Generation Complete!", "Completed:"),
        ("Generation Complete!",),
    ],
    'statistics': [
        ("Statistics Complete!", "Completed:"),
        ("All Steps Completed:",),
    ],
    # Add your custom markers here
}
```

Each entry is a tuple of strings that must ALL appear in the log for completion.

### Error Detection

The supervisor also detects errors via patterns:

```python
ERROR_PATTERNS = [
    r'(CUDA out of memory)',
    r'(RuntimeError:.*)',
    r'(FileNotFoundError:.*)',
    # Add custom error patterns here
]
```

## Result Aggregation

The supervisor automatically aggregates results into three CSV files:

### 1. `statistics_block_summary.csv`
Block-wise (prefix-length) comparison statistics with all 12 comparison types.

### 2. `statistics_decade_summary.csv`  
Decade-wise token frequencies (without comparison types or row_type column).

### 3. `token_counts_merged.csv`
Merged token counts with experiment metadata.

Each file includes columns for all experiment parameters:
- `n`, `c`, `h`, `g`, `k`, `t`, `model`, `dataset`

### Export Location
```
/projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/exports/
```

### Data Formatting for Export
- All-zero rows are dropped
- Zero values are replaced with empty strings (for sparsity)
- Each row clearly identifies the experiment

## Monitoring

### Dashboard

View the real-time dashboard:
```bash
cat ../supervisor_state/dashboard_v2.txt
```

The dashboard shows:
- Job status summary
- GPU slot usage
- Current GPU assignments
- Recent job activity
- Runtime configuration file locations

### State File

The supervisor state is persisted to JSON:
```bash
cat ../supervisor_state/pipeline_state_v2.json
```

This allows recovery after supervisor restart.

## Troubleshooting

### "Job disappeared from queue without completion markers"

This means the job finished but the supervisor couldn't confirm completion:

1. Check the job's log file for errors
2. Verify output files exist
3. Add the job's completion message to `COMPLETION_MARKERS`

### Jobs Not Starting

1. Check if GPU slots are available:
   ```bash
   cat ../supervisor_state/gpu_config.yaml
   ```

2. Check priority settings:
   ```yaml
   priority:
     max_priority_level: 5  # Increase to include more experiments
     min_priority: 0        # Lower to allow all priorities
   ```

### Modifying Running Experiments

To stop assigning jobs to a specific node:
1. Edit `gpu_config.yaml` and remove that node
2. Currently running jobs will complete
3. New jobs will use remaining nodes

## File Locations

| File | Purpose |
|------|---------|
| `supervisor_v2.py` | Main supervisor script |
| `supervisor_v2_config.yaml` | Main configuration |
| `supervisor_v2.sh` | SLURM job wrapper |
| `../supervisor_state/` | Runtime state directory |
| `../supervisor_state/gpu_config.yaml` | Runtime GPU configuration |
| `../supervisor_state/dashboard_v2.txt` | Human-readable status |
| `../supervisor_state/pipeline_state_v2.json` | Machine-readable state |
| `../supervisor_state/backups/` | Automatic backups |

## Datasets

### Creating Datasets

Use the dataset creation script:
```bash
python create_generative_datasets.py --config ../config/generative_datasets_config.yaml
```

### Dataset Definitions

| Dataset | Age at pos 1000 | Focus |
|---------|----------------|-------|
| GD0 | 1-30 | Childhood/Young |
| GD1 | 30-49 | Middle-age |
| GD2 | 50-69 | Late middle-age |
| GD3 | 70-99 | Old age |
| GD4 | Mixed | 20% 0-29, 25% 30-49, 25% 50-69, 20% 70-99, 10% death |

GDB0-GDB4 are the birthday-token versions (same row indices, different H5 file).

## Best Practices

1. **Start with High Priority** - Begin with `max_priority_level: 1` to run base experiments first
2. **Monitor Regularly** - Check the dashboard every few hours
3. **Backup Configs** - The supervisor backs itself up, but keep your own config backups
4. **Test Small** - Run with a single model/dataset first to verify setup
5. **Export Promptly** - Copy export CSVs as they're generated

## Quick Reference

```bash
# Submit supervisor
sbatch supervisor_v2.sh

# Check status
cat ../supervisor_state/dashboard_v2.txt

# View logs
tail -f /projects/0/prjs1589/stonybrook/logs/supervisor_v2-*.out

# Stop supervisor
scancel $(squeue -u $USER -n supervisor_v2 -h -o %i)

# Add GPU node at runtime
echo "gpus:
  ossc9424vm1: [0, 1, 2, 3]
  ossc9424vm2: [0, 1, 2, 3]" > ../supervisor_state/gpu_config.yaml

# Check exports
ls -la /projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/exports/
```
