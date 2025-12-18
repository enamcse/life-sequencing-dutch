# Trie Builder - Setup Complete! ✅

## Status

✅ **All files created and configured**
✅ **Job submitted and running** (Job ID: 17558409)
✅ **Output directory created**: `/projects/0/prjs1589/stonybrook/visualize/trie_tree/`

## What Was Fixed

1. **SLURM partition**: Changed from `thin` to `rome` (correct for your cluster)
2. **Output directory**: Changed to `/projects/0/prjs1589/stonybrook/visualize/trie_tree/`
3. **Test script**: Fixed import paths (though testing requires torch in venv)

## Current Job

```bash
Job ID: 17558409
Partition: rome
Status: RUNNING
Node: tcn376
```

## Monitor Your Job

### Check Status
```bash
squeue -u $USER
```

### View Logs (once they appear)
```bash
# Follow error log (detailed progress)
tail -f /projects/0/prjs1589/stonybrook/logs/build_trie-17558409.err

# View output log (high-level status)
tail -f /projects/0/prjs1589/stonybrook/logs/build_trie-17558409.out
```

### Check sacct (after completion)
```bash
sacct -j 17558409 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

## Expected Output Files

When the job completes, you'll find:

1. **`dryrun_trie.csv`** - Main trie structure
   - Location: `/projects/0/prjs1589/stonybrook/visualize/trie_tree/dryrun_trie.csv`
   - Format: node_id, token, parent, count, end_count, child_list

2. **`dryrun_trie_metadata.json`** - Job metadata
   - Configuration used
   - Statistics before/after pruning
   - Special token IDs

3. **Log files**
   - `/projects/0/prjs1589/stonybrook/logs/build_trie-17558409.err` - Detailed log
   - `/projects/0/prjs1589/stonybrook/logs/build_trie-17558409.out` - Summary

## After Job Completes

### 1. Check Output
```bash
# List output files
ls -lh /projects/0/prjs1589/stonybrook/visualize/trie_tree/

# View first few rows
head /projects/0/prjs1589/stonybrook/visualize/trie_tree/dryrun_trie.csv

# Count nodes
wc -l /projects/0/prjs1589/stonybrook/visualize/trie_tree/dryrun_trie.csv
```

### 2. View Metadata
```bash
cat /projects/0/prjs1589/stonybrook/visualize/trie_tree/dryrun_trie_metadata.json | python -m json.tool
```

### 3. Generate Visualization
```bash
cd ~/life-sequencing-dutch

python -m pop2vec.llm.src.new_code.visualize_trie \
    /projects/0/prjs1589/stonybrook/visualize/trie_tree/dryrun_trie.csv \
    /projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv \
    --output ~/trie_visualization.html \
    --title "Life Sequence Patterns (Dryrun Data)"
```

### 4. Download and View (on your local machine)
```bash
# On your local machine
scp snellius:~/trie_visualization.html .
firefox trie_visualization.html
```

## Troubleshooting

### If Job Fails

1. **Check logs**:
```bash
grep -i error /projects/0/prjs1589/stonybrook/logs/build_trie-17558409.err
tail -100 /projects/0/prjs1589/stonybrook/logs/build_trie-17558409.err
```

2. **Common issues**:
   - Out of memory → Increase `lower_limit` in config
   - File not found → Check paths in config
   - Timeout → Increase time limit in SLURM script

### If Log Files Don't Appear

SLURM may buffer output. The files will appear when:
- Job completes
- Buffer flushes
- Or use `--unbuffered` in Python script

## Configuration Reference

Current config (`pop2vec/llm/configs/Snellius/build_trie_config.json`):
```json
{
    "input_file": ".../dryrun_encoded.h5",
    "output_file": ".../visualize/trie_tree/dryrun_trie.csv",
    "vocab_file": ".../vocab_v0.csv",
    "lower_limit": 10,        // Min count to keep
    "max_nodes": 100000,      // Max nodes
    "max_sequences": null,    // Process all
    "skip_background": true,  // Skip demographics
    "max_seq_len": 512,
    "mlm_encoded": false
}
```

## Testing Locally (requires full environment)

The test script requires torch and other dependencies. To test:

1. **Load proper environment**:
```bash
cd ~/life-sequencing-dutch
source requirements/load_venv.sh
```

2. **Run test**:
```bash
python pop2vec/llm/src/new_code/test_trie.py
```

Note: This may not work in interactive node if modules aren't available. Better to test via SLURM.

## Quick Reference Commands

```bash
# Submit job
sbatch pop2vec/llm/slurm_scripts/snellius/build_trie.sh

# Check status
squeue -u $USER

# Cancel job (if needed)
scancel 17558409

# View logs
tail -f /projects/0/prjs1589/stonybrook/logs/build_trie-17558409.err

# Check output
ls -lh /projects/0/prjs1589/stonybrook/visualize/trie_tree/

# Visualize
python -m pop2vec.llm.src.new_code.visualize_trie \
    /projects/0/prjs1589/stonybrook/visualize/trie_tree/dryrun_trie.csv \
    /projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv \
    -o ~/trie_viz.html
```

## Next Steps

1. ⏳ **Wait for job to complete** (~10-30 minutes for dryrun data)
2. ✅ **Check output files**
3. 🎨 **Generate visualization**
4. 📊 **Analyze patterns!**

## Files Summary

All files are in place:

```
life-sequencing-dutch/pop2vec/llm/
├── src/new_code/
│   ├── build_sequence_trie.py       ✅ Main script
│   ├── visualize_trie.py             ✅ Visualization
│   └── test_trie.py                  ✅ Tests (fixed)
├── configs/Snellius/
│   └── build_trie_config.json        ✅ Config (updated path)
├── slurm_scripts/snellius/
│   └── build_trie.sh                 ✅ SLURM script (fixed partition)
├── scripts/
│   └── monitor_trie_job.sh           ✅ Monitor helper
├── TRIE_ANALYSIS.md                  ✅ Full documentation
├── TRIE_IMPLEMENTATION_SUMMARY.md    ✅ Technical details
└── TRIE_QUICK_REFERENCE.md           ✅ Quick guide
```

## Current Job Status

As of now:
- Job ID: 17558409
- Status: RUNNING
- Partition: rome
- Node: tcn376
- Runtime: ~2-3 minutes so far

**The job should complete successfully!** Check back in 10-30 minutes. 🚀
