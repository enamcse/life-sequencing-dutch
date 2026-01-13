#!/bin/bash
#SBATCH --job-name=mortality_labels
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH -p thin
#SBATCH -e /projects/0/prjs1589/stonybrook/logs/%x-%j.err
#SBATCH -o /projects/0/prjs1589/stonybrook/logs/%x-%j.out

# ============================================================================
# MORTALITY PREDICTION LABEL GENERATION - SLURM SCRIPT
# ============================================================================
# 
# This script generates mortality prediction labels from background and death
# registry data.
#
# Usage:
#   sbatch run_mortality_labels.sh
#
# Or with overrides:
#   sbatch --export=ALL,BACKGROUND_FILE=/custom/path/background.parquet run_mortality_labels.sh
#
# ============================================================================

# ============================================================================
# CONFIGURABLE PARAMETERS - EDIT THESE
# ============================================================================

# Input files (step 2 data)
BACKGROUND_FILE="${BACKGROUND_FILE:-/projects/0/prjs1589/stonybrook/cbs_data/step2/background.parquet}"
DEATH_FILE="${DEATH_FILE:-/projects/0/prjs1589/stonybrook/cbs_data/step2/death.parquet}"

# Output directory (parent of all, subset, all-splits, subset-splits)
# These folders must already exist!
OUTPUT_DIR="${OUTPUT_DIR:-/projects/0/prjs1589/stonybrook/evaluation/labels/mortality}"

# Statistics output directory
STATS_DIR="${STATS_DIR:-/projects/0/prjs1589/stonybrook/evaluation/labels/mortality/stats}"

# Random seed for reproducibility
RANDOM_SEED="${RANDOM_SEED:-42}"

# Subset size (max number of samples in subset)
SUBSET_SIZE="${SUBSET_SIZE:-200000}"

# Skip plot generation (set to 1 to skip)
SKIP_PLOTS="${SKIP_PLOTS:-0}"

# ============================================================================
# SCRIPT LOCATION
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/generate_mortality_labels.py"

# ============================================================================
# EXECUTION
# ============================================================================

echo "=========================================="
echo "MORTALITY PREDICTION LABEL GENERATION"
echo "=========================================="
echo "Started: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo ""
echo "Configuration:"
echo "  Background file: ${BACKGROUND_FILE}"
echo "  Death file: ${DEATH_FILE}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Statistics directory: ${STATS_DIR}"
echo "  Random seed: ${RANDOM_SEED}"
echo "  Subset size: ${SUBSET_SIZE}"
echo "  Skip plots: ${SKIP_PLOTS}"
echo ""
echo "Script: ${PYTHON_SCRIPT}"
echo "=========================================="
echo ""

# Validate input files
if [ ! -f "${BACKGROUND_FILE}" ]; then
    echo "ERROR: Background file not found: ${BACKGROUND_FILE}"
    exit 1
fi

if [ ! -f "${DEATH_FILE}" ]; then
    echo "ERROR: Death file not found: ${DEATH_FILE}"
    exit 1
fi

# Validate output folder structure
for folder in "all" "subset" "all-splits" "subset-splits"; do
    if [ ! -d "${OUTPUT_DIR}/${folder}" ]; then
        echo "ERROR: Required folder does not exist: ${OUTPUT_DIR}/${folder}"
        echo "Please create the folder structure first:"
        echo "  mkdir -p ${OUTPUT_DIR}/{all,subset,all-splits,subset-splits}"
        exit 1
    fi
done

# Create stats directory if needed
mkdir -p "${STATS_DIR}"

# Load environment
echo "Loading Python environment..."
cd ~/life-sequencing-dutch/
source requirements/load_venv.sh

# Build command
CMD="python ${PYTHON_SCRIPT} \
    --background-file ${BACKGROUND_FILE} \
    --death-file ${DEATH_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --stats-dir ${STATS_DIR} \
    --seed ${RANDOM_SEED} \
    --subset-size ${SUBSET_SIZE}"

if [ "${SKIP_PLOTS}" == "1" ]; then
    CMD="${CMD} --skip-plots"
fi

echo ""
echo "Running command:"
echo "${CMD}"
echo ""

# Execute
eval ${CMD}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "COMPLETED SUCCESSFULLY"
else
    echo "FAILED with exit code: ${EXIT_CODE}"
fi
echo "Finished: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
