#!/bin/bash
#SBATCH --job-name=h5_stats
#SBATCH --output=logs/h5_statistics_%j.out
#SBATCH --error=logs/h5_statistics_%j.err
#SBATCH --partition=thin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00

# ============================================================================
# H5 Sequence Statistics - SLURM Job Script
# ============================================================================
# Computes comprehensive statistics about sequence properties in HDF5 files.
# Uses vectorized numpy operations and parallel processing for speed.
#
# Usage:
#   sbatch run_h5_statistics.sh /path/to/encoded.h5
#   sbatch run_h5_statistics.sh /path/to/encoded.h5 /path/to/output_stats.txt
#
#   # With lifespan criteria check (generates config for extractor)
#   sbatch run_h5_statistics.sh /path/to/encoded.h5 /path/to/output_stats.txt lifespan
#
# Output:
#   - stats.txt: Human-readable statistics report
#   - stats_age_pairs.csv: (age_0, age_end) pair frequencies
#   - stats_decades.csv: Decade distributions at positions 0, 6, end_pos
#   - stats_lifespan_stats.csv: Lifespan criteria match counts (if --lifespan_check)
#   - lifespan_criteria.json: Config for extractor (if --generate_config)
# ============================================================================

set -e

# Default paths - EDIT THESE
DEFAULT_H5_FILE="/home/ehassan/life-sequencing-dutch/encoded.h5"
DEFAULT_OUTPUT_DIR="/home/ehassan/life-sequencing-dutch/pop2vec/llm/src/new_code/gen_eval/output"

# Parse arguments
H5_FILE="${1:-$DEFAULT_H5_FILE}"
OUTPUT_FILE="${2:-${DEFAULT_OUTPUT_DIR}/h5_statistics_$(date +%Y%m%d_%H%M%S).txt}"
MODE="${3:-basic}"  # basic, lifespan, or scan

# Number of workers - REDUCED to avoid OOM
N_WORKERS=${N_WORKERS:-8}

# Chunk size - REDUCED to be memory safe
CHUNK_SIZE=${CHUNK_SIZE:-100000}

# Sequential mode - set to true if you still get OOM errors
SEQUENTIAL=${SEQUENTIAL:-false}

# End position (default 1023)
END_POS=${END_POS:-1023}

# Scan range (e.g., "1000-1023")
SCAN_RANGE=${SCAN_RANGE:-}

echo "=============================================="
echo "H5 Sequence Statistics"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURM_NODELIST}"
echo "CPUs:          ${SLURM_CPUS_PER_TASK}"
echo "Memory:        ${SLURM_MEM_PER_NODE}"
echo "Start time:    $(date)"
echo ""
echo "Input file:    ${H5_FILE}"
echo "Output file:   ${OUTPUT_FILE}"
echo "Mode:          ${MODE}"
echo "Workers:       ${N_WORKERS}"
echo "Chunk size:    ${CHUNK_SIZE}"
echo "Sequential:    ${SEQUENTIAL}"
echo "End position:  ${END_POS}"
echo "=============================================="

# Create output directory if needed
mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p logs

# Verify input file exists
if [ ! -f "$H5_FILE" ]; then
    echo "ERROR: Input file not found: $H5_FILE"
    exit 1
fi

# Get file size for reference
FILE_SIZE=$(du -h "$H5_FILE" | cut -f1)
echo "Input file size: $FILE_SIZE"
echo ""

# Activate environment if needed (uncomment and modify as needed)
# source /path/to/venv/bin/activate
# module load python/3.10

# Set environment variables for performance
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1  # Prevent numpy from using too many threads per process

# Change to project directory
cd /home/ehassan/life-sequencing-dutch

echo "Starting statistics computation..."
echo ""

# Build command with optional flags
CMD="python -m pop2vec.llm.src.new_code.gen_eval.src.h5_sequence_statistics \
    --h5_file $H5_FILE \
    --output $OUTPUT_FILE \
    --n_workers $N_WORKERS \
    --chunk_size $CHUNK_SIZE \
    --end_pos $END_POS"

if [ "$SEQUENTIAL" = "true" ]; then
    CMD="$CMD --sequential"
fi

# Add mode-specific options
if [ "$MODE" = "lifespan" ]; then
    CONFIG_FILE="${OUTPUT_FILE%.txt}_criteria.json"
    CMD="$CMD --lifespan_check --generate_config $CONFIG_FILE"
    echo "Lifespan mode: will generate config at $CONFIG_FILE"
elif [ "$MODE" = "scan" ]; then
    if [ -n "$SCAN_RANGE" ]; then
        CMD="$CMD --scan_range $SCAN_RANGE --find_real_end"
        echo "Scan mode: scanning positions $SCAN_RANGE"
    else
        CMD="$CMD --scan_range 900-1023 --find_real_end"
        echo "Scan mode: scanning positions 900-1023 (default)"
    fi
fi

# Run the statistics script
eval $CMD

echo ""
echo "=============================================="
echo "Job completed at: $(date)"
echo "Output files:"
echo "  - ${OUTPUT_FILE}"
echo "  - ${OUTPUT_FILE%.txt}_age_pairs.csv"
echo "  - ${OUTPUT_FILE%.txt}_decades.csv"
if [ "$MODE" = "lifespan" ]; then
    echo "  - ${OUTPUT_FILE%.txt}_lifespan_stats.csv"
    echo "  - ${OUTPUT_FILE%.txt}_criteria.json (for use with extractor)"
fi
if [ "$MODE" = "scan" ]; then
    echo "  - ${OUTPUT_FILE%.txt}_position_scan.csv"
    echo "  - ${OUTPUT_FILE%.txt}_real_end_positions.csv"
fi
echo "=============================================="
