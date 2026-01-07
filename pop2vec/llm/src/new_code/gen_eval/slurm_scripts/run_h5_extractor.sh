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
# H5 Sequence Extractor - SLURM Job Script (Paired File Support)
# ============================================================================
# Extracts sequences matching specific criteria from HDF5 datasets.
# Supports paired extraction from two H5 files (original + birthday token).
# 
# The script applies criteria filtering on the PRIMARY file, then extracts
# the same indices from BOTH files. Verifies sequence_ids match between files.
#
# Features:
#   - Vectorized numpy operations for speed
#   - Parallel processing
#   - Paired file extraction with sequence_id verification
#   - Config-based position-age criteria
#
# Usage:
#   1. Edit the PARAMETERS section below
#   2. Run: sbatch run_h5_extractor.sh
#
# Or override via command line:
#   sbatch run_h5_extractor.sh generate_config
#   sbatch run_h5_extractor.sh [custom_h5_file] [custom_output] [n_sequences]
# ============================================================================

set -e

# ============================================================================
# PARAMETERS - EDIT THIS SECTION
# ============================================================================

# ----------------------------------------------------------------------------
# REQUIRED PARAMETERS
# ----------------------------------------------------------------------------

# Primary H5 file (criteria are applied on this file)
# Set to empty string "" or "None" to use command line argument
H5_FILE_PRIMARY="/home/ehassan/life-sequencing-dutch/encoded.h5"

# Secondary H5 file (birthday token version) - same indices extracted
# Set to empty string "" or "None" for single-file mode
H5_FILE_SECONDARY="/home/ehassan/life-sequencing-dutch/encoded_birthday.h5"

# Output directory
OUTPUT_DIR="/home/ehassan/life-sequencing-dutch/pop2vec/llm/src/new_code/gen_eval/output"

# Number of sequences to extract
N_SEQUENCES=10000

# ----------------------------------------------------------------------------
# OPTIONAL PARAMETERS
# ----------------------------------------------------------------------------

# Criteria mode: "config" or "legacy"
#   - config: Use a JSON config file with position-based age criteria
#   - legacy: Use named criteria (childhood_start,full_length,end_of_life)
CRITERIA_MODE="config"

# Config file path (used when CRITERIA_MODE="config")
# Set to empty string "" or "None" to use default lifespan criteria
CONFIG_FILE=""

# Legacy criteria (used when CRITERIA_MODE="legacy")
# Options: childhood_start, full_length, end_of_life, decade_70, decade_80, decade_90, all
LEGACY_CRITERIA="childhood_start,full_length,end_of_life"

# Number of parallel workers (reduce if OOM)
N_WORKERS=8

# Chunk size for parallel processing (reduce if OOM)
CHUNK_SIZE=100000

# Sequential mode (set to "true" if you still get OOM errors)
SEQUENTIAL="false"

# Random seed for reproducibility
SEED=42

# Skip sequence_id verification between paired files
SKIP_ID_VERIFICATION="false"

# Output file naming (timestamp will be appended)
OUTPUT_PREFIX="extracted"

# ============================================================================
# END OF PARAMETERS
# ============================================================================

# Helper function to check if a value is empty/None/null
is_empty() {
    local val="$1"
    if [ -z "$val" ] || [ "$val" = "None" ] || [ "$val" = "null" ] || [ "$val" = "none" ]; then
        return 0  # true, is empty
    fi
    return 1  # false, not empty
}

# Check for generate_config mode
if [ "$1" = "generate_config" ]; then
    CONFIG_PATH="${2:-${OUTPUT_DIR}/lifespan_criteria.json}"
    echo "Generating default lifespan config..."
    mkdir -p "$(dirname "$CONFIG_PATH")"
    cd /home/ehassan/life-sequencing-dutch
    python -m pop2vec.llm.src.new_code.gen_eval.src.h5_sequence_extractor --generate_config "$CONFIG_PATH"
    echo "Config saved to: $CONFIG_PATH"
    exit 0
fi

# Allow command line overrides
if [ -n "$1" ] && [ "$1" != "generate_config" ]; then
    H5_FILE_PRIMARY="$1"
fi
if [ -n "$2" ]; then
    OUTPUT_DIR="$(dirname "$2")"
    OUTPUT_PREFIX="$(basename "$2" .h5)"
fi
if [ -n "$3" ]; then
    N_SEQUENCES="$3"
fi

# Generate timestamp for output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Determine if paired mode
if is_empty "$H5_FILE_SECONDARY"; then
    PAIRED_MODE="false"
    OUTPUT_PRIMARY="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${TIMESTAMP}.h5"
    OUTPUT_SECONDARY=""
else
    PAIRED_MODE="true"
    OUTPUT_PRIMARY="${OUTPUT_DIR}/${OUTPUT_PREFIX}_original_${TIMESTAMP}.h5"
    OUTPUT_SECONDARY="${OUTPUT_DIR}/${OUTPUT_PREFIX}_birthday_${TIMESTAMP}.h5"
fi

echo "=============================================="
echo "H5 Sequence Extractor"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURM_NODELIST:-$(hostname)}"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Memory:        ${SLURM_MEM_PER_NODE:-N/A}"
echo "Start time:    $(date)"
echo ""
echo "--- INPUT FILES ---"
echo "Primary file:    ${H5_FILE_PRIMARY}"
if [ "$PAIRED_MODE" = "true" ]; then
    echo "Secondary file:  ${H5_FILE_SECONDARY}"
    echo "Paired mode:     ENABLED"
else
    echo "Paired mode:     DISABLED (single file)"
fi
echo ""
echo "--- OUTPUT FILES ---"
echo "Primary output:  ${OUTPUT_PRIMARY}"
if [ "$PAIRED_MODE" = "true" ]; then
    echo "Secondary output: ${OUTPUT_SECONDARY}"
fi
echo ""
echo "--- EXTRACTION SETTINGS ---"
echo "N sequences:     ${N_SEQUENCES}"
echo "Criteria mode:   ${CRITERIA_MODE}"
if [ "$CRITERIA_MODE" = "config" ]; then
    if is_empty "$CONFIG_FILE"; then
        echo "Config file:     (default lifespan criteria)"
    else
        echo "Config file:     ${CONFIG_FILE}"
    fi
else
    echo "Legacy criteria: ${LEGACY_CRITERIA}"
fi
echo ""
echo "--- PERFORMANCE SETTINGS ---"
echo "Workers:         ${N_WORKERS}"
echo "Chunk size:      ${CHUNK_SIZE}"
echo "Sequential:      ${SEQUENTIAL}"
echo "Seed:            ${SEED}"
if [ "$PAIRED_MODE" = "true" ]; then
    echo "Skip ID verify:  ${SKIP_ID_VERIFICATION}"
fi
echo "=============================================="

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Verify input files exist
if [ ! -f "$H5_FILE_PRIMARY" ]; then
    echo "ERROR: Primary input file not found: $H5_FILE_PRIMARY"
    exit 1
fi

if [ "$PAIRED_MODE" = "true" ] && [ ! -f "$H5_FILE_SECONDARY" ]; then
    echo "ERROR: Secondary input file not found: $H5_FILE_SECONDARY"
    exit 1
fi

# Get file sizes for reference
echo ""
echo "Input file sizes:"
FILE_SIZE_PRIMARY=$(du -h "$H5_FILE_PRIMARY" | cut -f1)
echo "  Primary:   $FILE_SIZE_PRIMARY"
if [ "$PAIRED_MODE" = "true" ]; then
    FILE_SIZE_SECONDARY=$(du -h "$H5_FILE_SECONDARY" | cut -f1)
    echo "  Secondary: $FILE_SIZE_SECONDARY"
fi
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

# Build the base command
CMD="python -m pop2vec.llm.src.new_code.gen_eval.src.h5_sequence_extractor \
    --h5_file $H5_FILE_PRIMARY \
    --output $OUTPUT_PRIMARY \
    --n_sequences $N_SEQUENCES \
    --n_workers $N_WORKERS \
    --chunk_size $CHUNK_SIZE \
    --seed $SEED"

# Add paired file arguments
if [ "$PAIRED_MODE" = "true" ]; then
    CMD="$CMD --h5_file_secondary $H5_FILE_SECONDARY"
    CMD="$CMD --output_secondary $OUTPUT_SECONDARY"
    
    if [ "$SKIP_ID_VERIFICATION" = "true" ]; then
        CMD="$CMD --skip_id_verification"
    fi
fi

# Add criteria arguments
if [ "$CRITERIA_MODE" = "config" ]; then
    if is_empty "$CONFIG_FILE"; then
        # Generate a temporary default config
        TEMP_CONFIG="${OUTPUT_DIR}/temp_lifespan_criteria_${TIMESTAMP}.json"
        echo "Generating default lifespan config..."
        python -m pop2vec.llm.src.new_code.gen_eval.src.h5_sequence_extractor --generate_config "$TEMP_CONFIG"
        CMD="$CMD --config $TEMP_CONFIG"
    else
        CMD="$CMD --config $CONFIG_FILE"
    fi
else
    CMD="$CMD --criteria $LEGACY_CRITERIA"
fi

# Add sequential flag if needed
if [ "$SEQUENTIAL" = "true" ]; then
    CMD="$CMD --sequential"
fi

# Run the extractor script
echo "Running command:"
echo "$CMD"
echo ""
eval $CMD

echo ""
echo "=============================================="
echo "Job completed at: $(date)"
echo ""
echo "Output files:"
echo "  Primary:   ${OUTPUT_PRIMARY}"
if [ -f "$OUTPUT_PRIMARY" ]; then
    OUTPUT_SIZE_PRIMARY=$(du -h "$OUTPUT_PRIMARY" | cut -f1)
    echo "             Size: $OUTPUT_SIZE_PRIMARY"
fi
echo "  Summary:   ${OUTPUT_PRIMARY%.h5}_summary.txt"

if [ "$PAIRED_MODE" = "true" ]; then
    echo ""
    echo "  Secondary: ${OUTPUT_SECONDARY}"
    if [ -f "$OUTPUT_SECONDARY" ]; then
        OUTPUT_SIZE_SECONDARY=$(du -h "$OUTPUT_SECONDARY" | cut -f1)
        echo "             Size: $OUTPUT_SIZE_SECONDARY"
    fi
    echo "  Paired summary: ${OUTPUT_PRIMARY%.h5}_paired_summary.txt"
fi

if [ "$CRITERIA_MODE" = "config" ]; then
    echo ""
    echo "  Criteria config: ${OUTPUT_PRIMARY%.h5}_criteria.json"
fi

echo "=============================================="
