#!/bin/bash
# SLURM Log Analyzer - Runner Script
# ===================================
# Run this script to analyze all SLURM logs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/analysis_output"
LOG_DIRS="/gpfs/ostor/ossc9424/logs3 /gpfs/ostor/ossc9424/logs2 /gpfs/ostor/ossc9424/logs"

# Parse arguments
WORKERS=${WORKERS:-8}
SAMPLE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --workers|-w)
            WORKERS="$2"
            shift 2
            ;;
        --sample|-s)
            SAMPLE="--sample $2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --workers, -w N    Number of parallel workers (default: 8)"
            echo "  --sample, -s N     Only process N files (for testing)"
            echo "  --output, -o DIR   Output directory"
            echo "  --help, -h         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "SLURM Log Analyzer"
echo "=============================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Workers: ${WORKERS}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# Step 1: Quick stats
echo "[Step 1/4] Running quick stats..."
python3 "${SCRIPT_DIR}/quick_stats.py" $LOG_DIRS 2>&1 | tee "${OUTPUT_DIR}/quick_stats.txt"
echo ""

# Step 2: Full analysis
echo "[Step 2/4] Running full log analysis..."
python3 "${SCRIPT_DIR}/analyze_slurm_logs.py" \
    --dirs $LOG_DIRS \
    --output "${OUTPUT_DIR}" \
    --workers "${WORKERS}" \
    ${SAMPLE}
echo ""

# Step 3: Pretraining analysis
echo "[Step 3/4] Analyzing pretraining logs..."
python3 "${SCRIPT_DIR}/analyze_pretrain.py" \
    --dirs $LOG_DIRS \
    --output "${OUTPUT_DIR}/pretrain_detailed.json" \
    --workers "${WORKERS}"
echo ""

# Step 4: Evaluation results extraction
echo "[Step 4/4] Extracting evaluation results..."
python3 "${SCRIPT_DIR}/extract_eval_results.py" \
    --dirs $LOG_DIRS \
    --output "${OUTPUT_DIR}/eval_results" \
    --workers "${WORKERS}"
echo ""

echo "=============================================="
echo "Analysis Complete!"
echo "=============================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Key files:"
echo "  ${OUTPUT_DIR}/summary_report.txt    - Human-readable summary"
echo "  ${OUTPUT_DIR}/monthly_analysis.json - Jobs by month"
echo "  ${OUTPUT_DIR}/job_type_analysis.json - Jobs by type"
echo "  ${OUTPUT_DIR}/pretrain_detailed.json - Pretraining details"
echo "  ${OUTPUT_DIR}/eval_results.json     - Evaluation results"
echo "  ${OUTPUT_DIR}/eval_results.csv      - Evaluation results (CSV)"
echo "=============================================="
