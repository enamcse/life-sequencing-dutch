#!/bin/bash
# Quick diagnostic for trie job

JOBID="17560862"
ERR_FILE="/projects/0/prjs1589/stonybrook/logs/build_trie_parallel-${JOBID}.err"
OUT_FILE="/projects/0/prjs1589/stonybrook/logs/build_trie_parallel-${JOBID}.out"

echo "=========================================="
echo "TRIE JOB DIAGNOSTIC"
echo "=========================================="
echo "Job ID: $JOBID"
echo "Time: $(date)"
echo ""

# Check if job is running
echo "=== JOB STATUS ==="
squeue -j $JOBID -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R" || echo "Job not in queue"
echo ""

# Check file sizes and modification times
echo "=== LOG FILES ==="
ls -lh "$ERR_FILE" "$OUT_FILE" 2>/dev/null || echo "Log files not found"
echo ""

echo "=== FILE MODIFICATION TIMES ==="
echo "Error log: $(stat -c '%y' $ERR_FILE 2>/dev/null || echo 'Not found')"
echo "Output log: $(stat -c '%y' $OUT_FILE 2>/dev/null || echo 'Not found')"
echo ""

# Calculate age
if [ -f "$ERR_FILE" ]; then
    AGE=$(($(date +%s) - $(stat -c %Y "$ERR_FILE")))
    echo "Error log last modified: ${AGE} seconds ago"
    echo ""
fi

# Show recent content
echo "=== RECENT ERROR LOG (last 30 lines) ==="
tail -30 "$ERR_FILE" 2>/dev/null || echo "Cannot read error log"
echo ""

echo "=== RECENT OUTPUT LOG (last 20 lines) ==="
tail -20 "$OUT_FILE" 2>/dev/null || echo "Cannot read output log"
echo ""

# Check for worker progress
WORKER_MSGS=$(grep -c "Worker.*:" "$OUT_FILE" 2>/dev/null || echo "0")
echo "=== WORKER PROGRESS MESSAGES ===""
echo "Found $WORKER_MSGS worker progress messages"
if [ "$WORKER_MSGS" -gt "0" ]; then
    echo "Latest worker updates:"
    grep "Worker.*:" "$OUT_FILE" | tail -5
fi
echo ""

# Decision
echo "=== DIAGNOSIS ==="
if [ "$WORKER_MSGS" -gt "10" ]; then
    echo "✅ Job is WORKING - workers are reporting progress"
    echo "   Action: Keep waiting, monitor progress"
elif [ "$AGE" -lt "300" ]; then
    echo "⏳ Job recently active but no worker reports yet"
    echo "   Action: Wait 5-10 more minutes"
elif [ "$AGE" -lt "600" ]; then
    echo "⚠️  No updates in 5-10 minutes"
    echo "   Action: Wait 5 more minutes OR cancel with: scancel $JOBID"
else
    echo "❌ No updates in 10+ minutes - likely STUCK"
    echo "   Action: Cancel with: scancel $JOBID"
    echo "   Then: Use fewer workers or single-threaded version"
fi
echo ""

echo "=========================================="
echo "COMMANDS:"
echo "  Cancel job:    scancel $JOBID"
echo "  Watch errors:  tail -f $ERR_FILE"
echo "  Watch output:  tail -f $OUT_FILE"
echo "=========================================="
