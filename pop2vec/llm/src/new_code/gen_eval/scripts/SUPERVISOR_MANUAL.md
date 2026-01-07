# Pipeline Supervisor - User Manual

A fully automated SLURM-based supervisor for managing large-scale generative evaluation experiments. Designed to run in air-gapped environments without internet access.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)
7. [Architecture](#architecture)

---

## Overview

### The Problem
You have:
- **5 models** to evaluate
- **4 experiments** per model (20 total generation jobs)
- **4 H100 GPUs** on 1-2 nodes
- Jobs can take hours; PuTTY connections drop

### The Solution
A **supervisor job** that runs on the work node (`ossc9424vm0`) and:
1. Submits generation jobs to GPUs with collision avoidance
2. Monitors completion via SLURM queue and output files
3. Auto-submits statistics jobs when generation finishes
4. Handles failures with optional auto-retry
5. Writes a human-readable dashboard every 60 seconds

```
┌─────────────────────────────────────────────────────────────┐
│  ossc9424vm0 (work_env) - Supervisor Job                    │
│  ─────────────────────────────────────────────────────────  │
│  While not all jobs done:                                   │
│    1. squeue → update job states                            │
│    2. Submit new gen jobs to free GPU slots                 │
│    3. Submit stats jobs for completed generations           │
│    4. Write dashboard.txt and state.json                    │
│    5. Sleep 60 seconds                                      │
└──────────────────────────┬──────────────────────────────────┘
                           │ sbatch
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ossc9424vm1 (comp_env) - GPU Jobs                          │
│  GPU0: gen_model1_exp1 → gen_model2_exp1 → ...              │
│  GPU1: gen_model1_exp2 → gen_model3_exp1 → ...              │
│  GPU2: gen_model1_exp3 → ...                                │
│  GPU3: gen_model1_exp4 → ...                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Step 1: Prepare SLURM Scripts

First, generate SLURM scripts for all model/experiment combinations:

```bash
cd ~/life-sequencing-dutch/pop2vec/llm/src/new_code/gen_eval/scripts

# Generate scripts with GPU assignments
python generate_slurm.py \
    --config ../config/experiments_config.yaml \
    --gpus "ossc9424vm1:0,1,2,3"
```

This creates files like:
- `slurm_scripts/gen_model_v1_exp_n10_c100_h20_g100.sh`
- `slurm_scripts/stats_model_v1_exp_n10_c100_h20_g100.sh`

### Step 2: Configure the Supervisor

Edit `supervisor_config.yaml`:

```yaml
models:
  - model_v1_gen_20251117
  - model_v2
  - model_v3
  - model_v4
  - model_v5

experiments:
  - exp_n10_c100_h20_g100
  - exp_n100_c100_h20_g100
  - exp_n100_c1000_h20_g100
  - exp_n1000_c100_h20_g100

gpus:
  ossc9424vm1:
    - 0
    - 1
    - 2
    - 3
```

### Step 3: Submit the Supervisor

```bash
sbatch supervisor.slurm
```

You'll see: `Submitted batch job 12345`

### Step 4: Monitor Progress

```bash
# View the dashboard (updates every 60 seconds)
watch -n 10 cat ../supervisor_state/dashboard.txt

# Or just once:
cat ../supervisor_state/dashboard.txt
```

### Step 5: Stop When Done (or if needed)

```bash
# Find supervisor job ID
squeue -u $USER | grep supervisor

# Cancel it
scancel 12345
```

---

## Configuration

### `supervisor_config.yaml` Options

| Option | Default | Description |
|--------|---------|-------------|
| `models` | (required) | List of model names |
| `experiments` | (required) | List of experiment names |
| `gpus` | ossc9424vm1: [0,1,2,3] | GPU slots per node |
| `poll_interval_seconds` | 60 | How often to check status |
| `auto_submit_stats` | true | Submit stats after gen completes |
| `auto_resubmit_failed` | false | Retry failed jobs |
| `max_retries` | 2 | Max retry attempts |

### GPU Configuration

**Single node (4 GPUs):**
```yaml
gpus:
  ossc9424vm1:
    - 0
    - 1
    - 2
    - 3
```

**Two nodes (8 GPUs total):**
```yaml
gpus:
  ossc9424vm1:
    - 0
    - 1
    - 2
    - 3
  ossc9424vm2:
    - 0
    - 1
    - 2
    - 3
```

---

## Usage

### Workflow Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Generation  │ ──▶ │  Statistics  │ ──▶ │    Plots     │
│  (GPU, slow) │     │  (CPU, fast) │     │  (CPU, fast) │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Commands

```bash
# Submit supervisor (starts everything)
sbatch supervisor.slurm

# Check your jobs
squeue -u $USER

# View dashboard
cat ../supervisor_state/dashboard.txt

# View detailed state
cat ../supervisor_state/pipeline_state.json | python -m json.tool

# Stop supervisor
scancel <supervisor_job_id>

# Check GPU job logs
tail -f /projects/0/prjs1589/stonybrook/logs/gen_model_v1*-*.out

# Check supervisor logs
tail -f /projects/0/prjs1589/stonybrook/logs/supervisor-*.out
```

### Resuming After Interruption

The supervisor saves state to `pipeline_state.json`. If the supervisor is cancelled or crashes, simply resubmit:

```bash
sbatch supervisor.slurm
```

It will:
1. Load the saved state
2. Skip already-completed jobs
3. Resume from where it left off

---

## Monitoring

### Dashboard Output

The supervisor writes `supervisor_state/dashboard.txt` every 60 seconds:

```
================================================================================
PIPELINE SUPERVISOR DASHBOARD
Updated: 2026-01-07 14:30:00
================================================================================

GENERATION JOBS:
  Total: 20
  ✓ Completed: 8
  ⟳ Running:   4
  ⏳ Queued:    4
  ✗ Failed:    0
  - Pending:   4

STATISTICS JOBS:
  Total: 20
  ✓ Completed: 6
  ⟳ Running:   2
  ⏳ Queued:    0
  ✗ Failed:    0
  - Pending:   12

GPU SLOTS:
  ossc9424vm1:0: gen_model1_exp2 (running)
  ossc9424vm1:1: gen_model2_exp1 (running)
  ossc9424vm1:2: gen_model3_exp1 (running)
  ossc9424vm1:3: gen_model4_exp1 (running)

JOB MATRIX (Generation):
--------------------------------------------------------------------------------
Model                          n10_c100     n100_c100    n100_c1000   n1000_c100
--------------------------------------------------------------------------------
model_v1_gen_20251117           ✓ 0:45:23    ✓ 2:15:00    ⟳            -
model_v2                        ✓ 0:42:10    ⟳            -            -
model_v3                        ⟳            -            -            -
model_v4                        ⟳            -            -            -
model_v5                        -            -            -            -

TIMING STATISTICS:
--------------------------------------------------------------------------------
  Completed Gen Jobs: 8
  Average Duration: 1:30:45
  Min Duration: 0:42:10
  Max Duration: 2:15:00
  Remaining Jobs: 12
  Est. Time (with 4 GPUs): 4:32:15

================================================================================
```

### Checking Progress via `check_progress.py`

You can also use the standalone progress checker:

```bash
# Simple view
python check_progress.py

# With statistics
python check_progress.py --stats

# Specific models/experiments
python check_progress.py -m model_v1 model_v2 -e exp_n10_c100_h20_g100
```

---

## Troubleshooting

### Supervisor Won't Start

```bash
# Check supervisor log
cat /projects/0/prjs1589/stonybrook/logs/supervisor-*.err

# Common issues:
# - supervisor_config.yaml not found
# - Python environment not loaded
# - Missing dependencies
```

### Jobs Stuck in PENDING

```bash
# Check why jobs are pending
squeue -u $USER -t PENDING --format="%.18i %.30j %.8T %.10M %.9l %.6D %R"

# Common reasons:
# - Partition resources unavailable
# - Dependency on previous job not met
```

### Jobs Keep Failing

1. Check the job's log file:
```bash
cat /projects/0/prjs1589/stonybrook/logs/gen_model_v1_exp_n100_c100_h20_g100-*.err
```

2. Common issues:
   - **CUDA out of memory**: Reduce `generation_batch_size` in experiment config
   - **File not found**: Check checkpoint/data paths
   - **Module not found**: Environment not loaded correctly

3. Enable auto-retry in config:
```yaml
auto_resubmit_failed: true
max_retries: 2
```

### Resetting the Pipeline

To start fresh:

```bash
# Cancel all your jobs
scancel -u $USER

# Remove state files
rm -rf ../supervisor_state/

# Optionally remove output files
# rm -rf /projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/*/exp_*/

# Resubmit
sbatch supervisor.slurm
```

### Manual Job Submission

If you need to submit a specific job manually:

```bash
# Submit a generation job
sbatch ../slurm_scripts/gen_model_v1_exp_n10_c100_h20_g100.sh

# Submit with dependency (wait for job 12345)
sbatch --dependency=afterok:12345 ../slurm_scripts/stats_model_v1_exp_n10_c100_h20_g100.sh
```

---

## Architecture

### Files Created

```
gen_eval/
├── scripts/
│   ├── supervisor.py           # Main supervisor code
│   ├── supervisor.slurm        # SLURM job to run supervisor
│   ├── supervisor_config.yaml  # Configuration
│   ├── generate_slurm.py       # Generates SLURM scripts
│   ├── submit_jobs.sh          # Manual job submission
│   └── check_progress.py       # Progress checker
├── slurm_scripts/
│   ├── gen_model_v1_exp_n10_c100_h20_g100.sh
│   ├── stats_model_v1_exp_n10_c100_h20_g100.sh
│   └── ...
├── supervisor_state/
│   ├── dashboard.txt           # Human-readable status
│   └── pipeline_state.json     # Machine-readable state
└── config/
    ├── experiments_config.yaml
    └── models_config.yaml
```

### State Persistence

The supervisor saves its state to `pipeline_state.json`:

```json
{
  "models": ["model_v1", "model_v2"],
  "experiments": ["exp_n10_c100_h20_g100"],
  "jobs": {
    "gen_model_v1_exp_n10_c100_h20_g100": {
      "status": "completed",
      "slurm_job_id": "12345",
      "duration_seconds": 3600.5,
      ...
    }
  },
  "gpu_assignments": {
    "ossc9424vm1:0": "gen_model_v2_exp_n10_c100_h20_g100"
  }
}
```

### GPU Collision Avoidance

The supervisor tracks which GPU slot (node:index) is assigned to which job. When submitting:

1. Find GPU slots without running/queued jobs
2. Submit new job to that slot
3. Add `--dependency=afterany:<prev_job>` if a job previously used that slot

This ensures jobs on the same GPU run sequentially.

---

## Example: Full Run with 5 Models × 4 Experiments

```bash
# 1. Setup models (one time)
python setup_model.py --config ../config/models_config.yaml

# 2. Generate all SLURM scripts
python generate_slurm.py \
    --config ../config/experiments_config.yaml \
    --gpus "ossc9424vm1:0,1,2,3"

# 3. Edit supervisor config
vim supervisor_config.yaml
# Add all 5 models and 4 experiments

# 4. Submit supervisor
sbatch supervisor.slurm

# 5. Monitor (in a separate terminal or check periodically)
watch -n 30 cat ../supervisor_state/dashboard.txt

# 6. When done, check final statistics
python check_progress.py --stats
```

Expected runtime for 20 generation jobs with 4 GPUs:
- If each job takes ~2 hours average
- 20 jobs / 4 GPUs = 5 "batches"
- Total: ~10 hours

The supervisor will keep running until all jobs complete, then exit.

---

## Questions?

Check the logs:
```bash
# Supervisor log
tail -100 /projects/0/prjs1589/stonybrook/logs/supervisor-*.out

# Specific job log
ls -la /projects/0/prjs1589/stonybrook/logs/gen_*
```
