#!/bin/bash
# Helper script to monitor trie building job

JOB_NAME="build_trie"
LOG_DIR="/projects/0/prjs1589/stonybrook/logs"

echo "=========================================="
echo "TRIE BUILDER JOB MONITOR"
echo "=========================================="
echo ""

# Check for running jobs
echo "Checking for running jobs..."
JOBS=$(squeue -u $USER -n $JOB_NAME --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" | tail -n +2)

if [ -z "$JOBS" ]; then
    echo "No running jobs found."
    echo ""
    echo "Checking recent completed jobs..."
    # Look for recent log files
    RECENT_LOGS=$(ls -t $LOG_DIR/${JOB_NAME}-*.err 2>/dev/null | head -3)
    
    if [ -z "$RECENT_LOGS" ]; then
        echo "No recent log files found."
        echo ""
        echo "To submit a new job:"
        echo "  sbatch pop2vec/llm/slurm_scripts/snellius/build_trie.sh"
    else
        echo "Recent job logs:"
        for log in $RECENT_LOGS; do
            JOBID=$(basename $log | sed 's/.*-\([0-9]*\)\.err/\1/')
            echo ""
            echo "Job ID: $JOBID"
            echo "  Error log: $log"
            echo "  Output log: ${log%.err}.out"
            
            # Check if completed successfully
            if grep -q "TRIE BUILDING COMPLETE" $log 2>/dev/null; then
                echo "  Status: ✓ COMPLETED"
                # Get output file
                OUTPUT=$(grep "Saved trie to:" $log 2>/dev/null | tail -1 | awk '{print $NF}')
                if [ -n "$OUTPUT" ]; then
                    echo "  Output: $OUTPUT"
                    if [ -f "$OUTPUT" ]; then
                        ROWS=$(wc -l < "$OUTPUT")
                        echo "  Size: $((ROWS-1)) nodes"
                    fi
                fi
            elif grep -q "error\|Error\|ERROR\|failed\|Failed\|FAILED" $log 2>/dev/null; then
                echo "  Status: ✗ FAILED"
                echo "  Last error:"
                grep -i "error\|failed" $log | tail -3 | sed 's/^/    /'
            else
                echo "  Status: ? UNKNOWN (check logs)"
            fi
        done
    fi
else
    echo "Running jobs:"
    echo "$JOBS"
    echo ""
    
    # Get job ID
    JOBID=$(echo "$JOBS" | awk '{print $1}' | head -1)
    
    echo "Job ID: $JOBID"
    echo "Error log: $LOG_DIR/${JOB_NAME}-${JOBID}.err"
    echo "Output log: $LOG_DIR/${JOB_NAME}-${JOBID}.out"
    echo ""
    
    # Check if log files exist yet
    if [ -f "$LOG_DIR/${JOB_NAME}-${JOBID}.err" ]; then
        echo "Recent log output:"
        echo "----------------------------------------"
        tail -20 "$LOG_DIR/${JOB_NAME}-${JOBID}.err"
        echo "----------------------------------------"
        echo ""
        echo "To follow logs in real-time:"
        echo "  tail -f $LOG_DIR/${JOB_NAME}-${JOBID}.err"
    else
        echo "Log files not created yet (job may be queued or just started)"
        echo ""
        echo "Wait a moment and try:"
        echo "  tail -f $LOG_DIR/${JOB_NAME}-${JOBID}.err"
    fi
fi

echo ""
echo "=========================================="
