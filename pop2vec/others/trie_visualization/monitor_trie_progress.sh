#!/bin/bash
#
# Real-time progress monitor for trie building
# Shows actual worker progress, not just completed workers
#

JOBID="17560983"
OUT_FILE="/projects/0/prjs1589/stonybrook/logs/build_trie_parallel-${JOBID}.out"

echo "=========================================="
echo "REAL-TIME TRIE BUILDER PROGRESS"
echo "=========================================="
echo "Job ID: $JOBID"
echo "Time: $(date)"
echo ""

# Check job status
echo "=== JOB STATUS ==="
squeue -j $JOBID -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R" 2>/dev/null || echo "Job not in queue"
echo ""

# Extract worker progress
echo "=== WORKER PROGRESS ==="
if [ -f "$OUT_FILE" ]; then
    # Get latest progress from each worker
    echo "Latest progress reports:"
    grep "Worker.*:" "$OUT_FILE" | tail -20
    echo ""
    
    # Calculate overall progress
    TOTAL_REPORTS=$(grep -c "Worker.*%.*nodes" "$OUT_FILE" 2>/dev/null || echo "0")
    
    if [ "$TOTAL_REPORTS" -gt "0" ]; then
        # Get the most recent percentage from any worker
        LATEST_PCT=$(grep "Worker.*%.*nodes" "$OUT_FILE" | tail -1 | grep -oP '\(\K[0-9.]+(?=%\))')
        
        echo "=========================================="
        echo "Overall Progress:"
        echo "  Latest worker: ~${LATEST_PCT}% complete"
        echo "  Progress reports: $TOTAL_REPORTS"
        echo ""
        
        # Estimate time remaining
        if [ -f "$OUT_FILE" ]; then
            START_TIME=$(stat -c %Y "$OUT_FILE")
            CURRENT_TIME=$(date +%s)
            ELAPSED=$((CURRENT_TIME - START_TIME))
            
            if (( $(echo "$LATEST_PCT > 5" | bc -l) )); then
                TOTAL_EST=$(echo "scale=0; $ELAPSED * 100 / $LATEST_PCT" | bc)
                REMAINING=$((TOTAL_EST - ELAPSED))
                
                echo "  Time elapsed: $((ELAPSED / 60)) minutes"
                echo "  Estimated remaining: $((REMAINING / 60)) minutes"
                echo "  Estimated total: $((TOTAL_EST / 60)) minutes"
            else
                echo "  Time elapsed: $((ELAPSED / 60)) minutes"
                echo "  (Too early for time estimate)"
            fi
        fi
    else
        echo "No progress reports yet (workers still initializing)"
    fi
else
    echo "Output file not found: $OUT_FILE"
fi

echo ""
echo "=========================================="
echo "NOTE: The 'Processing chunks: 0/16' bar"
echo "shows COMPLETED workers, not active progress."
echo "Watch the worker reports above for real progress!"
echo "=========================================="
echo ""
echo "To monitor continuously:"
echo "  watch -n 10 'bash $0'"
echo "  tail -f $OUT_FILE"
