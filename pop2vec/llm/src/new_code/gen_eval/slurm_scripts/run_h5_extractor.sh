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
# Default criteria (all three must be met):
#   1. Childhood start: age at index 6 is in 0-9
#   2. Full length: token/age at index 1023 is non-zero
#   3. End-of-life: age at index 1023 is in 70-99
#
# Usage:
#   sbatch run_h5_extractor.sh /path/to/encoded.h5 /path/to/output.h5 10000
#   sbatch run_h5_extractor.sh /path/to/encoded.h5 /path/to/output.h5 10000 "childhood_start,full_length,decade_80"
#
# Output:
#   - extracted.h5: HDF5 file with extracted sequences
#   - extracted_summary.txt: Extraction summary and statistics
# ============================================================================

set -e

# Default paths - EDIT THESE
DEFAULT_H5_FILE="/home/ehassan/life-sequencing-dutch/encoded.h5"
DEFAULT_OUTPUT_DIR="/home/ehassan/life-sequencing-dutch/pop2vec/llm/src/new_code/gen_eval/output"

# Parse arguments
H5_FILE="${1:-$DEFAULT_H5_FILE}"
OUTPUT_FILE="${2:-${DEFAULT_OUTPUT_DIR}/extracted_$(date +%Y%m%d_%H%M%S).h5}"
N_SEQUENCES="${3:-10000}"
CRITERIA="${4:-childhood_start,full_length,end_of_life}"

# Number of workers (use most of available CPUs)
N_WORKERS=${SLURM_CPUS_PER_TASK:-32}
N_WORKERS=$((N_WORKERS - 2))  # Leave 2 CPUs for system overhead

# Chunk size - larger chunks = faster but more memory
CHUNK_SIZE=500000

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
echo "Criteria:      ${CRITERIA}"
echo "Workers:       ${N_WORKERS}"
echo "Chunk size:    ${CHUNK_SIZE}"
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

# Run the extractor script
python -m pop2vec.llm.src.new_code.gen_eval.src.h5_sequence_extractor \
    --h5_file "$H5_FILE" \
    --output "$OUTPUT_FILE" \
    --n_sequences "$N_SEQUENCES" \
    --criteria "$CRITERIA" \
    --n_workers "$N_WORKERS" \
    --chunk_size "$CHUNK_SIZE" \
    --seed "$SEED"

echo ""
echo "=============================================="
echo "Job completed at: $(date)"
echo "Output files:"
echo "  - ${OUTPUT_FILE}"
echo "  - ${OUTPUT_FILE%.h5}_summary.txt"

# Show output file size
if [ -f "$OUTPUT_FILE" ]; then
    OUTPUT_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "Output file size: $OUTPUT_SIZE"
fi
echo "=============================================="
