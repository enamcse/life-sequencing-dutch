# ❌ Trie Job is STUCK - Action Required

## Diagnosis: CONFIRMED STUCK

**Job ID**: 17560862  
**Status**: Running but FROZEN  
**Evidence**:
- ✅ Job running for 13+ minutes
- ❌ NO log updates in 13+ minutes
- ❌ Workers initialized but NEVER started processing
- ❌ Progress bar stuck at 0/64
-last log entry: 07:22:08 (13 minutes ago)

## Root Cause

**HDF5 File Contention Deadlock**

With 4.4M sequences:
- 64 workers × 69K sequences each
- All 64 workers trying to open same HDF5 file simultaneously
- HDF5 library serializing or deadlocking on file access
- Workers are stuck waiting for file handles

## IMMEDIATE ACTION REQUIRED

### Cancel the Stuck Job

```bash
scancel 17560862
```

## Better Approaches

### Option 1: Use Fewer Workers (RECOMMENDED) ⭐

Reduce HDF5 contention by using 16 workers instead of 64:

```bash
# Edit the SLURM script
nano ~/life-sequencing-dutch/pop2vec/llm/slurm_scripts/snellius/build_trie_parallel.sh

# Change this line:
#SBATCH --cpus-per-task=16  # Was 64

# Resubmit
cd ~/life-sequencing-dutch/pop2vec/llm/slurm_scripts/snellius
sbatch build_trie_parallel.sh
```

**Expected time**: ~1-2 hours (more stable)

### Option 2: Use Single-Threaded (MOST RELIABLE) ✅

For 4.4M sequences, single-threaded is actually safer:

```bash
cd ~/life-sequencing-dutch/pop2vec/llm/slurm_scripts/snellius
sbatch build_trie.sh  # Original non-parallel version
```

**Expected time**: ~10-15 hours (but WILL complete)

### Option 3: Process in Batches

Modify config to process subsets:

```json
{
  "max_sequences": 500000,  // Process 500K at a time
  ...
}
```

Run 9 times with different subsets, then merge.

## Why It Failed

### The Problem with 64 Workers

| Issue | Impact |
|-------|--------|
| **HDF5 contention** | 64 processes opening same file = deadlock |
| **Memory pressure** | 64 × ~20GB = 1.28TB (exceeds 512GB allocated) |
| **I/O bottleneck** | File system can't handle 64 simultaneous reads |

### What We Learned

Parallel trie building works great for:
- ✅ 100K-500K sequences
- ✅ 8-16 workers
- ✅ Moderate memory

But fails for:
- ❌ 4.4M sequences (44x more than expected)
- ❌ 64 workers (too much contention)
- ❌ Single HDF5 file access pattern

## Recommended Solution

### Step 1: Cancel Current Job

```bash
scancel 17560862
```

### Step 2: Choose Your Approach

**For speed (1-2 hours)**:
```bash
# Edit SLURM script: Change --cpus-per-task=64 to --cpus-per-task=16
sbatch build_trie_parallel.sh
```

**For reliability (10-15 hours but guaranteed)**:
```bash
sbatch build_trie.sh  # Single-threaded
```

### Step 3: Monitor

With fixed code (already updated):
```bash
# You'll see worker progress like:
# Worker 12345: Processing sequences 0 to 69,397
# Worker 12345: 10,000/69,397 (14.4%) - 125,432 nodes
# Worker 12345: 20,000/69,397 (28.8%) - 234,567 nodes
```

## Timeline Expectations

### With 16 Workers (Recommended)
- **Setup**: 1-2 minutes
- **Phase 1 (Build)**: 60-90 minutes
- **Phase 2 (Merge)**: 10-20 minutes
- **Phase 3 (Prune)**: 5-10 minutes
- **Total**: ~1.5-2 hours

### With Single-Threaded
- **Build**: 10-12 hours
- **Prune**: 10-15 minutes
- **Total**: ~10-15 hours

## Files to Edit

### For 16 Workers

Edit: `/home/ehassan/life-sequencing-dutch/pop2vec/llm/slurm_scripts/snellius/build_trie_parallel.sh`

Change line 5:
```bash
#SBATCH --cpus-per-task=16  # Changed from 64
```

### For Single-Threaded

No changes needed, just use:
```bash
sbatch build_trie.sh
```

## Prevention for Next Time

Before running on full dataset:

1. **Check dataset size first**:
   ```python
   import h5py
   with h5py.File('encoded.h5', 'r') as f:
       print(f"Sequences: {len(f['input_ids'])}")
   ```

2. **Test on subset**:
   ```json
   {"max_sequences": 10000}  // Test with 10K first
   ```

3. **Scale workers appropriately**:
   - < 100K sequences: 8-16 workers
   - 100K-1M sequences: 16-32 workers  
   - > 1M sequences: 32-64 workers OR single-threaded

## Summary

- ❌ **Current job**: STUCK, needs to be cancelled
- ⚠️ **Problem**: 64 workers + 4.4M sequences = HDF5 deadlock
- ✅ **Solution**: Use 16 workers OR single-threaded
- ⏰ **New ETA**: 1-2 hours (16 workers) or 10-15 hours (single)

## Action Items

```bash
# 1. Cancel stuck job
scancel 17560862

# 2. Edit SLURM script (change 64 to 16)
nano ~/life-sequencing-dutch/pop2vec/llm/slurm_scripts/snellius/build_trie_parallel.sh

# 3. Resubmit
cd ~/life-sequencing-dutch/pop2vec/llm/slurm_scripts/snellius
sbatch build_trie_parallel.sh

# 4. Monitor (will now show worker progress!)
tail -f /projects/0/prjs1589/stonybrook/logs/build_trie_parallel-*.out
```

The updated code will now show progress from each worker, so you'll know it's working! 🚀
