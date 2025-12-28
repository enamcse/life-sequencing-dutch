#!/bin/bash
#
#SBATCH --job-name=log_analysis
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH -p rome
#SBATCH -e /projects/0/prjs1589/stonybrook/logs/%x-%j.err
#SBATCH -o /projects/0/prjs1589/stonybrook/logs/%x-%j.out

# =============================================================================
# SLURM Log Analyzer - Cluster Job Script
# =============================================================================
# This script runs the log analysis on the cluster using SLURM.
# It's designed for processing the full ~8GB of logs efficiently.
#
# Submit with: sbatch run_analysis_slurm.sh
# =============================================================================

echo "========================================"
echo "SLURM Log Analysis Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo ""

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load Python environment if needed
# Uncomment and modify if you need a specific environment
# source /path/to/your/venv/bin/activate
# module load Python/3.9.6-GCCcore-11.2.0

# Configuration
OUTPUT_DIR="${SCRIPT_DIR}/analysis_output"
LOG_DIRS="/gpfs/ostor/ossc9424/logs3 /gpfs/ostor/ossc9424/logs2 /gpfs/ostor/ossc9424/logs"
WORKERS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p "${OUTPUT_DIR}"

echo "Output directory: ${OUTPUT_DIR}"
echo "Workers: ${WORKERS}"
echo ""

# Step 1: Quick stats (fast overview)
echo "[Step 1/5] Running quick stats..."
python3 quick_stats.py $LOG_DIRS 2>&1 | tee "${OUTPUT_DIR}/quick_stats.txt"
echo ""

# Step 2: Full analysis
echo "[Step 2/5] Running full log analysis..."
python3 analyze_slurm_logs.py \
    --dirs $LOG_DIRS \
    --output "${OUTPUT_DIR}" \
    --workers "${WORKERS}"
echo ""

# Step 3: Pretraining analysis
echo "[Step 3/5] Analyzing pretraining logs..."
python3 analyze_pretrain.py \
    --dirs $LOG_DIRS \
    --output "${OUTPUT_DIR}/pretrain_detailed.json" \
    --workers "${WORKERS}"
echo ""

# Step 4: Evaluation results extraction
echo "[Step 4/5] Extracting evaluation results..."
python3 extract_eval_results.py \
    --dirs $LOG_DIRS \
    --output "${OUTPUT_DIR}/eval_results" \
    --workers "${WORKERS}"
echo ""

# Step 5: Generate HTML report
echo "[Step 5/5] Generating HTML report..."
python3 generate_html_report.py \
    --input "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/report.html"
echo ""

echo "========================================"
echo "Analysis Complete!"
echo "========================================"
echo "Ended: $(date)"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Key files:"
echo "  ${OUTPUT_DIR}/summary_report.txt     - Human-readable summary"
echo "  ${OUTPUT_DIR}/report.html            - Interactive HTML report"
echo "  ${OUTPUT_DIR}/monthly_analysis.json  - Jobs by month"
echo "  ${OUTPUT_DIR}/job_type_analysis.json - Jobs by type"
echo "  ${OUTPUT_DIR}/pretrain_detailed.json - Pretraining details"
echo "  ${OUTPUT_DIR}/eval_results.json      - Evaluation results"
echo "  ${OUTPUT_DIR}/eval_results.csv       - Evaluation results (CSV)"
echo "========================================"
