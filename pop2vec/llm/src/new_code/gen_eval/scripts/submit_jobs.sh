#!/bin/bash
#
# Submit SLURM Jobs for Generative Evaluation
#
# Features:
#   - Submit jobs for a specific experiment
#   - GPU-aware submission with sequential dependencies on same GPU
#   - Statistics jobs depend on their corresponding generation jobs
#   - Support for specifying unused GPU indices (optionally per VM)
#
# Usage:
#   bash submit_jobs.sh --experiment exp_n10_c100
#   bash submit_jobs.sh --experiment exp_n10_c100 --gpus 0,1,2
#   bash submit_jobs.sh --experiment exp_n10_c100 --gpus "ossc9424vm1:0,1,2;ossc9424vm2:0,1"
#   bash submit_jobs.sh --experiment exp_n10_c100 --gen-only
#

set -e

# Defaults
EXPERIMENT=""
PARTITION=""
GEN_ONLY=false
STATS_ONLY=false
DRY_RUN=false
GPU_INDICES=""  # Can be "0,1,2" or "vm1:0,1;vm2:0,1,2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_DIR="$SCRIPT_DIR/../slurm_scripts"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment|-e)
            EXPERIMENT="$2"
            shift 2
            ;;
        --partition|-p)
            PARTITION="$2"
            shift 2
            ;;
        --gpus|-g)
            GPU_INDICES="$2"
            shift 2
            ;;
        --gen-only)
            GEN_ONLY=true
            shift
            ;;
        --stats-only)
            STATS_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --slurm-dir)
            SLURM_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --experiment EXP_NAME [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --experiment, -e  Experiment name (required)"
            echo "  --gpus, -g        GPU indices to use for sequential dependencies"
            echo "                    Simple: '0,1,2' (default VM: ossc9424vm1)"
            echo "                    Per-VM: 'ossc9424vm1:0,1,2;ossc9424vm2:0,1'"
            echo "  --partition, -p   Override SLURM partition"
            echo "  --gen-only        Only submit generation jobs"
            echo "  --stats-only      Only submit statistics jobs"
            echo "  --dry-run         Print commands without executing"
            echo "  --slurm-dir       Directory containing SLURM scripts"
            echo "  --help, -h        Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 -e exp_n10_c100"
            echo "  $0 -e exp_n10_c100 --gpus 0,1,2"
            echo "  $0 -e exp_n10_c100 --gpus 'ossc9424vm1:0,1;ossc9424vm2:2,3'"
            echo "  $0 -e exp_n10_c100 --dry-run"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$EXPERIMENT" ]]; then
    echo "Error: --experiment is required"
    exit 1
fi

echo "=========================================="
echo "Submitting Jobs for Experiment: $EXPERIMENT"
echo "=========================================="
echo ""

# Find all scripts for this experiment
GEN_SCRIPTS=$(ls "$SLURM_DIR"/gen_*_"$EXPERIMENT".sh 2>/dev/null || true)
STATS_SCRIPTS=$(ls "$SLURM_DIR"/stats_*_"$EXPERIMENT".sh 2>/dev/null || true)

if [[ -z "$GEN_SCRIPTS" && -z "$STATS_SCRIPTS" ]]; then
    echo "No scripts found for experiment: $EXPERIMENT"
    echo "Looking in: $SLURM_DIR"
    exit 1
fi

# Parse GPU indices from manifest if available
MANIFEST="$SLURM_DIR/manifest_$EXPERIMENT.yaml"
if [[ -f "$MANIFEST" ]]; then
    echo "Using manifest: $MANIFEST"
fi

# Associative arrays to track job IDs
declare -A GEN_JOB_IDS       # Script name -> Job ID (for stats dependencies)
declare -A GPU_LAST_JOB_IDS  # "node:gpu_index" -> Last job ID (for sequential execution)

# Function to get GPU index from script (by reading the "GPU Index:" line from script)
get_gpu_index_from_script() {
    local script="$1"
    local gpu_idx=$(grep -o 'GPU Index: [0-9-]*' "$script" 2>/dev/null | grep -o '[0-9-]*' | head -1)
    echo "${gpu_idx:--1}"
}

# Function to get node name from script (by reading the "Node:" line from script - the config one, not the $(hostname))
get_node_from_script() {
    local script="$1"
    # Look for "Node: ossc9424vm1" pattern (the configured one, not the runtime $(hostname))
    local node=$(grep '^echo "Node: ' "$script" 2>/dev/null | head -1 | sed 's/echo "Node: //' | sed 's/"$//')
    echo "${node:-unknown}"
}

# Function to get GPU slot key (node:gpu_index) for tracking dependencies
get_gpu_slot_key() {
    local script="$1"
    local node=$(get_node_from_script "$script")
    local gpu_idx=$(get_gpu_index_from_script "$script")
    echo "${node}:${gpu_idx}"
}

# Submit generation jobs
if [[ "$STATS_ONLY" != "true" && -n "$GEN_SCRIPTS" ]]; then
    echo "Generation Jobs:"
    echo "-----------------"
    for script in $GEN_SCRIPTS; do
        script_basename=$(basename "$script")
        # Create the corresponding stats script name by replacing gen_ with stats_
        stats_script_name=${script_basename/gen_/stats_}
        
        # Get GPU slot key (node:gpu_index) for dependency tracking
        GPU_SLOT=$(get_gpu_slot_key "$script")
        GPU_IDX=$(get_gpu_index_from_script "$script")
        NODE=$(get_node_from_script "$script")
        
        # Build dependency argument (for sequential GPU execution on same node:gpu)
        DEP_ARG=""
        if [[ "$GPU_IDX" != "-1" && -n "${GPU_LAST_JOB_IDS[$GPU_SLOT]:-}" ]]; then
            DEP_ARG="--dependency=afterany:${GPU_LAST_JOB_IDS[$GPU_SLOT]}"
        fi
        
        if [[ "$DRY_RUN" == "true" ]]; then
            if [[ -n "$DEP_ARG" ]]; then
                echo "[DRY RUN] sbatch $DEP_ARG $script"
            else
                echo "[DRY RUN] sbatch $script"
            fi
            echo "  GPU Slot: $GPU_SLOT"
            # Use a fake job ID for dry run
            GEN_JOB_IDS["$stats_script_name"]="DRY_RUN_ID"
            if [[ "$GPU_IDX" != "-1" ]]; then
                GPU_LAST_JOB_IDS["$GPU_SLOT"]="DRY_RUN_ID"
            fi
        else
            echo "Submitting: $script_basename ($GPU_SLOT)"
            if [[ -n "$DEP_ARG" ]]; then
                echo "  Depends on job: ${GPU_LAST_JOB_IDS[$GPU_SLOT]}"
                JOB_ID=$(sbatch $DEP_ARG "$script" | awk '{print $4}')
            else
                JOB_ID=$(sbatch "$script" | awk '{print $4}')
            fi
            echo "  Job ID: $JOB_ID"
            GEN_JOB_IDS["$stats_script_name"]="$JOB_ID"
            if [[ "$GPU_IDX" != "-1" ]]; then
                GPU_LAST_JOB_IDS["$GPU_SLOT"]="$JOB_ID"
            fi
        fi
    done
    echo ""
fi

# Submit statistics jobs (with dependency on generation jobs if applicable)
if [[ "$GEN_ONLY" != "true" && -n "$STATS_SCRIPTS" ]]; then
    echo "Statistics Jobs:"
    echo "-----------------"
    for script in $STATS_SCRIPTS; do
        script_basename=$(basename "$script")
        
        # Check if we have a corresponding generation job ID
        DEP_ARG=""
        if [[ -n "${GEN_JOB_IDS[$script_basename]:-}" && "${GEN_JOB_IDS[$script_basename]}" != "DRY_RUN_ID" ]]; then
            DEP_ARG="--dependency=afterok:${GEN_JOB_IDS[$script_basename]}"
            echo "Submitting: $script_basename (depends on gen job ${GEN_JOB_IDS[$script_basename]})"
        else
            echo "Submitting: $script_basename"
        fi
        
        if [[ "$DRY_RUN" == "true" ]]; then
            if [[ -n "$DEP_ARG" ]]; then
                echo "[DRY RUN] sbatch $DEP_ARG $script"
            else
                echo "[DRY RUN] sbatch $script"
            fi
        else
            if [[ -n "$DEP_ARG" ]]; then
                JOB_ID=$(sbatch $DEP_ARG "$script" | awk '{print $4}')
            else
                JOB_ID=$(sbatch "$script" | awk '{print $4}')
            fi
            echo "  Job ID: $JOB_ID"
        fi
    done
    echo ""
fi

echo "=========================================="
echo "Submission complete!"
echo ""
echo "Check job status with: squeue -u \$USER"
echo "Check progress with: python check_progress.py -e $EXPERIMENT"
echo "=========================================="
