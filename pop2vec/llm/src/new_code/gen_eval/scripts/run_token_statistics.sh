#!/bin/bash
#SBATCH --job-name=token_stats
#SBATCH --output=logs/token_stats_%j.out
#SBATCH --error=logs/token_stats_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# ==============================================================================
# Dataset Token Statistics & Embeddings SLURM Script
# ==============================================================================
#
# This script runs the token statistics pipeline with three optional layers:
#   Layer 1: Compute token statistics (n_people, n_observation) from HDF5 files
#   Layer 2: Extract 2D PCA embeddings from pretrained model checkpoints
#   Layer 3: Generate t-SNE visualizations for each model
#
# Usage:
#   # Run all layers
#   sbatch run_token_statistics.slurm
#
#   # Run only token stats
#   sbatch --export=MODE=stats run_token_statistics.slurm
#
#   # Run only embeddings
#   sbatch --export=MODE=embeddings run_token_statistics.slurm
#
#   # Run only t-SNE
#   sbatch --export=MODE=tsne run_token_statistics.slurm
#
# ==============================================================================

# Print job information
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPUs: $SLURM_GPUS"
echo "Started: $(date)"
echo "=============================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# -----------------------------------------------------------------------------
# CONFIGURATION - MODIFY THESE PATHS
# -----------------------------------------------------------------------------

# Base directories
PROJECT_DIR="/home/ehassan/life-sequencing-dutch"
SCRIPT_DIR="${PROJECT_DIR}/pop2vec/llm/src/new_code/gen_eval"
PYTHON_SCRIPT="${SCRIPT_DIR}/src/dataset_token_statistics.py"

# Config file (YAML)
CONFIG_FILE="${SCRIPT_DIR}/config/datasets_config.yaml"

# Output directory
OUTPUT_DIR="${SCRIPT_DIR}/output/token_stats_$(date +%Y%m%d)"

# Python environment - activate your conda/venv environment here
# Uncomment and modify as needed:
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate myenv
# OR
# source /path/to/venv/bin/activate

# Number of parallel workers for HDF5 processing
N_WORKERS=16

# Chunk size for processing (larger = more memory, but faster)
CHUNK_SIZE=50000

# Minimum n_people for exportable tokens
MIN_PEOPLE=10

# t-SNE perplexity (5-50 typical, see script docstring for guidance)
TSNE_PERPLEXITY=30

# PAD token ID (usually 0)
PAD_ID=0

# -----------------------------------------------------------------------------
# RUN MODE SELECTION
# -----------------------------------------------------------------------------

# Determine which mode to run
MODE="${MODE:-all}"  # Default to "all" if not set

echo "Run mode: $MODE"
echo ""

case "$MODE" in
    "stats")
        MODE_FLAG="--only_token_stats"
        echo "Running Layer 1: Token Statistics only"
        ;;
    "embeddings")
        MODE_FLAG="--only_embeddings"
        echo "Running Layer 2: Model Embeddings only"
        ;;
    "tsne")
        MODE_FLAG="--only_tsne"
        echo "Running Layer 3: t-SNE Visualization only"
        ;;
    *)
        MODE_FLAG=""
        echo "Running all layers: Token Stats, Embeddings, t-SNE"
        ;;
esac

# -----------------------------------------------------------------------------
# CREATE OUTPUT DIRECTORY
# -----------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# -----------------------------------------------------------------------------
# RUN THE PIPELINE
# -----------------------------------------------------------------------------

echo "Starting pipeline..."
echo ""

python "${PYTHON_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_workers ${N_WORKERS} \
    --chunk_size ${CHUNK_SIZE} \
    --min_people ${MIN_PEOPLE} \
    --tsne_perplexity ${TSNE_PERPLEXITY} \
    --pad_id ${PAD_ID} \
    ${MODE_FLAG}

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=============================================="

# List output files
echo ""
echo "Output files:"
ls -lh "${OUTPUT_DIR}/"

exit ${EXIT_CODE}
