#!/bin/bash
#
# Submit SLURM Jobs for Generative Evaluation
#
# Usage:
#   bash submit_jobs.sh --experiment exp_n10_c100
#   bash submit_jobs.sh --experiment exp_n10_c100 --partition gpu_h100 --gen-only
#   bash submit_jobs.sh --experiment exp_n10_c100 --stats-only
#

set -e

# Defaults
EXPERIMENT=""
PARTITION=""
GEN_ONLY=false
STATS_ONLY=false
DRY_RUN=false
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
            echo "  --partition, -p   Override SLURM partition"
            echo "  --gen-only        Only submit generation jobs"
            echo "  --stats-only      Only submit statistics jobs"
            echo "  --dry-run         Print commands without executing"
            echo "  --slurm-dir       Directory containing SLURM scripts"
            echo "  --help, -h        Show this help"
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

# Associative array to track generation job IDs by script name
declare -A GEN_JOB_IDS

# Submit generation jobs
if [[ "$STATS_ONLY" != "true" && -n "$GEN_SCRIPTS" ]]; then
    echo "Generation Jobs:"
    echo "-----------------"
    for script in $GEN_SCRIPTS; do
        script_basename=$(basename "$script")
        # Create the corresponding stats script name by replacing gen_ with stats_
        stats_script_name=${script_basename/gen_/stats_}
        
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] sbatch $script"
            # Use a fake job ID for dry run
            GEN_JOB_IDS["$stats_script_name"]="DRY_RUN_ID"
        else
            echo "Submitting: $script_basename"
            JOB_ID=$(sbatch "$script" | awk '{print $4}')
            echo "  Job ID: $JOB_ID"
            GEN_JOB_IDS["$stats_script_name"]="$JOB_ID"
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
echo "=========================================="
