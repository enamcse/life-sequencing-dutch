#!/bin/bash
#SBATCH --job-name=h5_extract
#SBATCH --output=logs/h5_extractor_%j.out
#SBATCH --error=logs/h5_extractor_%j.err
#SBATCH --partition=thin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00

# ============================================================================
# H5 Sequence Extractor - SLURM Job Script
# ============================================================================
# Extracts sequences matching specific criteria from large HDF5 datasets.
# Uses vectorized numpy operations and parallel processing for speed.
#
# Two modes:
#   1. Legacy mode: Use named criteria (childhood_start, full_length, end_of_life)
#   2. Config mode: Use JSON config file with position-based age criteria
#
# Default lifespan criteria (config mode):
#   - Position 6: age 0-9 (childhood)
#   - Position 100: age 10-19 (teens)
#   - Position 200: age 20-29 (twenties)
#   - ... through ...
#   - Position 900: age 90-99 (nineties)
#   - Position 1000: age 90-99 (nineties_end)
#
# Usage:
#   # Legacy mode
#   sbatch run_h5_extractor.sh /path/to/encoded.h5 /path/to/output.h5 10000
#   sbatch run_h5_extractor.sh /path/to/encoded.h5 /path/to/output.h5 10000 "childhood_start,full_length"
#
#   # Config mode (recommended)
#   sbatch run_h5_extractor.sh /path/to/encoded.h5 /path/to/output.h5 10000 config /path/to/criteria.json
#
#   # Generate default config only
#   sbatch run_h5_extractor.sh generate_config /path/to/criteria.json
#
# Output:
#   - extracted.h5: HDF5 file with extracted sequences
#   - extracted_summary.txt: Extraction summary and statistics
#   - extracted_criteria.json: Copy of criteria used (config mode)
# ============================================================================

set -e

# Default paths - EDIT THESE
DEFAULT_H5_FILE="/home/ehassan/life-sequencing-dutch/encoded.h5"
DEFAULT_OUTPUT_DIR="/home/ehassan/life-sequencing-dutch/pop2vec/llm/src/new_code/gen_eval/output"

# Check for generate_config mode
if [ "$1" = "generate_config" ]; then
    CONFIG_PATH="${2:-${DEFAULT_OUTPUT_DIR}/lifespan_criteria.json}"
    echo "Generating default lifespan config..."
    cd /home/ehassan/life-sequencing-dutch
    python -m pop2vec.llm.src.new_code.gen_eval.src.h5_sequence_extractor --generate_config "$CONFIG_PATH"
    echo "Config saved to: $CONFIG_PATH"
    exit 0
fi

# Parse arguments
H5_FILE="${1:-$DEFAULT_H5_FILE}"
OUTPUT_FILE="${2:-${DEFAULT_OUTPUT_DIR}/extracted_$(date +%Y%m%d_%H%M%S).h5}"
N_SEQUENCES="${3:-10000}"
MODE_OR_CRITERIA="${4:-childhood_start,full_length,end_of_life}"
CONFIG_FILE="${5:-}"

# Number of workers - REDUCED to avoid OOM
N_WORKERS=${N_WORKERS:-8}

# Chunk size - REDUCED to be memory safe
CHUNK_SIZE=${CHUNK_SIZE:-100000}

# Sequential mode - set to true if you still get OOM errors
SEQUENTIAL=${SEQUENTIAL:-false}

# Random seed for reproducibility
SEED=42

echo "=============================================="
echo "H5 Sequence Extractor"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURM_NODELIST}"
echo "CPUs:          ${SLURM_CPUS_PER_TASK}"
echo "Memory:        ${SLURM_MEM_PER_NODE}"
echo "Start time:    $(date)"
echo ""
echo "Input file:    ${H5_FILE}"
echo "Output file:   ${OUTPUT_FILE}"
echo "N sequences:   ${N_SEQUENCES}"
echo "Mode/Criteria: ${MODE_OR_CRITERIA}"
if [ -n "$CONFIG_FILE" ]; then
    echo "Config file:   ${CONFIG_FILE}"
fi
echo "Workers:       ${N_WORKERS}"
echo "Chunk size:    ${CHUNK_SIZE}"
echo "Sequential:    ${SEQUENTIAL}"
echo "Seed:          ${SEED}"
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

echo "Starting sequence extraction..."
echo ""

# Build command based on mode
if [ "$MODE_OR_CRITERIA" = "config" ]; then
    # Config mode - use JSON config file
    if [ -z "$CONFIG_FILE" ] || [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Config mode requires a valid config file path as 5th argument"
        echo "Usage: sbatch run_h5_extractor.sh <h5_file> <output> <n_sequences> config <config.json>"
        exit 1
    fi
    
    CMD="python -m pop2vec.llm.src.new_code.gen_eval.src.h5_sequence_extractor \
        --h5_file $H5_FILE \
        --output $OUTPUT_FILE \
        --n_sequences $N_SEQUENCES \
        --config $CONFIG_FILE \
        --n_workers $N_WORKERS \
        --chunk_size $CHUNK_SIZE \
        --seed $SEED"
else
    # Legacy mode - use named criteria
    CMD="python -m pop2vec.llm.src.new_code.gen_eval.src.h5_sequence_extractor \
        --h5_file $H5_FILE \
        --output $OUTPUT_FILE \
        --n_sequences $N_SEQUENCES \
        --criteria $MODE_OR_CRITERIA \
        --n_workers $N_WORKERS \
        --chunk_size $CHUNK_SIZE \
        --seed $SEED"
fi

if [ "$SEQUENTIAL" = "true" ]; then
    CMD="$CMD --sequential"
fi

# Run the extractor script
eval $CMD

echo ""
echo "=============================================="
echo "Job completed at: $(date)"
echo "Output files:"
echo "  - ${OUTPUT_FILE}"
echo "  - ${OUTPUT_FILE%.h5}_summary.txt"
if [ "$MODE_OR_CRITERIA" = "config" ]; then
    echo "  - ${OUTPUT_FILE%.h5}_criteria.json"
fi

# Show output file size
if [ -f "$OUTPUT_FILE" ]; then
    OUTPUT_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "Output file size: $OUTPUT_SIZE"
fi
echo "=============================================="
