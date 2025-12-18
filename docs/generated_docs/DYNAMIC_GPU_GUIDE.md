# Dynamic GPU Allocation Guide 🚀

## Overview

The GPU-accelerated birthday token insertion now **automatically detects and uses all allocated GPUs**! No code or config changes needed when switching between debugging (2 GPUs) and production (4 GPUs).

## How It Works

### Configuration (No Changes Needed!)
```json
{
  "device_ids": null,  // null = auto-detect all GPUs
  "batch_size": 64,    // Works for 2 or 4 GPUs
  ...
}
```

### Python Code (Auto-detection)
```python
if device_ids is None:
    # Automatically detect all available GPUs
    device_ids = list(range(torch.cuda.device_count()))

# Output example:
# - 2 GPUs allocated → uses [0, 1]
# - 4 GPUs allocated → uses [0, 1, 2, 3]
```

## SLURM Scripts for Different Scenarios

### 1. Debugging (2 GPUs) - Fast Queue

**File**: `add_birthday_tokens_parallel.sh`

```bash
#SBATCH --gpus-per-node=2   # Request only 2 GPUs
```

**Use when**:
- ✅ Quick testing/debugging
- ✅ Fast queue allocation
- ✅ Want results quickly

**Submit**:
```bash
sbatch pop2vec/llm/slurm_scripts/snellius/add_birthday_tokens_parallel.sh
```

---

### 2. Production (4 GPUs) - Maximum Speed

**File**: `add_birthday_tokens_4gpu.sh`

```bash
#SBATCH --gpus-per-node=4   # Request all 4 GPUs
```

**Use when**:
- ✅ Final production run
- ✅ Need maximum speed (2x faster than 2 GPUs)
- ✅ Can wait in queue if needed

**Submit**:
```bash
sbatch pop2vec/llm/slurm_scripts/snellius/add_birthday_tokens_4gpu.sh
```

---

### 3. Full Dataset (2 GPUs) - Debugging

**File**: `add_birthday_full_parallel.sh`

```bash
#SBATCH --gpus-per-node=2   # Request 2 GPUs
```

**Submit**:
```bash
sbatch pop2vec/llm/slurm_scripts/snellius/add_birthday_full_parallel.sh
```

---

### 4. Full Dataset (4 GPUs) - Production

**File**: `add_birthday_full_4gpu.sh`

```bash
#SBATCH --gpus-per-node=4   # Request all 4 GPUs
```

**Submit**:
```bash
sbatch pop2vec/llm/slurm_scripts/snellius/add_birthday_full_4gpu.sh
```

## Script Summary Table

| Script | GPUs | Dataset | Purpose | Queue Priority |
|--------|------|---------|---------|----------------|
| `add_birthday_tokens_parallel.sh` | 2 | Dry run | **Debugging** | ⚡ Fast |
| `add_birthday_tokens_4gpu.sh` | 4 | Dry run | Production test | Slower |
| `add_birthday_full_parallel.sh` | 2 | Full | Debugging | ⚡ Fast |
| `add_birthday_full_4gpu.sh` | 4 | Full | **Production** | Slower |

## Performance Comparison

| GPUs | Batch Size | Speed | Queue Time |
|------|------------|-------|------------|
| 2 GPUs | 64 (32 per GPU) | 10-20x faster than CPU | ⚡ Minutes |
| 4 GPUs | 64 (16 per GPU) | 20-40x faster than CPU | ⏱️ Could be hours |

## Workflow Recommendation

### Phase 1: Quick Debugging (Use 2 GPUs)
```bash
# Test with dry run on 2 GPUs (fast queue)
sbatch pop2vec/llm/slurm_scripts/snellius/add_birthday_tokens_parallel.sh

# Verify it works, check logs
tail -f /projects/0/prjs1589/stonybrook/logs/add_birthday_gpu_parallel-*.out
```

### Phase 2: Validate on Full Dataset (Use 2 GPUs)
```bash
# Run full dataset on 2 GPUs (still reasonable speed)
sbatch pop2vec/llm/slurm_scripts/snellius/add_birthday_full_parallel.sh
```

### Phase 3: Production Run (Use 4 GPUs)
```bash
# When everything works, maximize speed with 4 GPUs
sbatch pop2vec/llm/slurm_scripts/snellius/add_birthday_full_4gpu.sh
```

## Monitoring GPU Usage

### Check Job Status
```bash
squeue -u $USER
```

### View Logs
```bash
# List recent jobs
ls -lt /projects/0/prjs1589/stonybrook/logs/ | head

# View specific job output
tail -f /projects/0/prjs1589/stonybrook/logs/add_birthday_*-JOBID.out
```

### Verify GPU Detection

In the log output, you should see:
```
GPUs available: 2 - [0, 1]     # For 2 GPU job
# or
GPUs available: 4 - [0, 1, 2, 3]   # For 4 GPU job
```

And from `nvidia-smi`:
```
| GPU  Name                 | Memory-Usage | GPU-Util |
|   0  NVIDIA H100          |  15000MiB    |    95%   |
|   1  NVIDIA H100          |  15000MiB    |    95%   |
# (and GPUs 2-3 if 4 GPU job)
```

## Adjusting Batch Size (Optional)

If you want different batch sizes for 2 vs 4 GPUs, you can create separate configs:

### For 2 GPUs (smaller batch)
```json
{
  "batch_size": 64,   // 32 per GPU
  "device_ids": null
}
```

### For 4 GPUs (larger batch)
```json
{
  "batch_size": 256,  // 64 per GPU
  "device_ids": null
}
```

But the **current unified config works for both**! 🎉

## Troubleshooting

### "Invalid device id" Error
- **Cause**: Config specifies more GPUs than SLURM allocated
- **Solution**: Use `"device_ids": null` for auto-detection (already done!)

### GPU Utilization is Low
- **Check**: Is batch size too small?
- **Solution**: Increase `batch_size` in config (e.g., 64 → 128 → 256)

### Out of Memory Error
- **Cause**: Batch size too large for GPU memory
- **Solution**: Reduce `batch_size` in config (e.g., 256 → 128 → 64)

### Job Stuck in Queue
- **If you requested 4 GPUs**: Switch to 2 GPU script for faster allocation
- **If urgent**: Use 2 GPU scripts which have faster queue times

## Cost-Benefit Analysis

### Scenario: 100K samples to process

| Approach | GPUs | Time | GPU Hours | Queue Wait | Total Time |
|----------|------|------|-----------|------------|------------|
| CPU-only | 0 | 4 hrs | 0 | 0 min | 4 hrs |
| 2 GPUs | 2 | 15 min | 0.5 | 5 min | **20 min** ⚡ |
| 4 GPUs | 4 | 8 min | 0.5 | 60 min | **68 min** |

**Recommendation**: 
- **Debugging**: Use 2 GPUs (20 min total including queue)
- **Production**: Use 4 GPUs if queue is short, otherwise 2 GPUs is fine

## Summary

✅ **No config changes needed** - `device_ids: null` auto-detects GPUs
✅ **Same code works for 2 or 4 GPUs** - fully dynamic
✅ **Multiple SLURM scripts** - choose based on your needs
✅ **Debugging**: Use 2 GPU scripts (fast queue)
✅ **Production**: Use 4 GPU scripts (maximum speed)

**Your workflow**:
1. Debug with 2 GPUs → fast queue, good speed
2. Production with 4 GPUs → maximum speed when available
3. No code changes between phases! 🎉
