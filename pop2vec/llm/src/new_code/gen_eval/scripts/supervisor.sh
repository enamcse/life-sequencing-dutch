#!/bin/bash
#SBATCH --job-name=supervisor
#SBATCH --partition=work_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH -o /projects/0/prjs1589/stonybrook/logs/supervisor-%j.out
#SBATCH -e /projects/0/prjs1589/stonybrook/logs/supervisor-%j.err

#
# Pipeline Supervisor SLURM Job
#
# This job runs continuously on the login node (work_env partition)
# and manages the entire generation/statistics pipeline.
#
# Features:
#   - Monitors SLURM queue for job status
#   - Submits generation jobs with GPU collision avoidance
#   - Automatically submits statistics jobs when generation completes
#   - Writes human-readable dashboard every 60 seconds
#   - Persists state to JSON for recovery
#
# Usage:
#   1. Edit supervisor_config.yaml with your models/experiments
#   2. Submit: sbatch supervisor.slurm
#   3. Monitor: cat supervisor_state/dashboard.txt
#   4. Stop: scancel <job_id>
#

echo "=========================================="
echo "Pipeline Supervisor"
echo "=========================================="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Script directory: $SCRIPT_DIR"
echo "Config file: $SCRIPT_DIR/supervisor_config.yaml"
echo ""

# Load environment
echo "Loading environment..."
cd ~/life-sequencing-dutch/
source requirements/load_venv.sh
cd "$SCRIPT_DIR"

# Check config exists
if [[ ! -f "supervisor_config.yaml" ]]; then
    echo "ERROR: supervisor_config.yaml not found!"
    echo "Please create it first. See supervisor_config.yaml.example"
    exit 1
fi

echo ""
echo "Starting supervisor..."
echo "Dashboard will be written to: $SCRIPT_DIR/../supervisor_state/dashboard.txt"
echo ""
echo "=========================================="

# Run supervisor
python supervisor.py --config supervisor_config.yaml

echo ""
echo "=========================================="
echo "Supervisor finished: $(date)"
echo "=========================================="
