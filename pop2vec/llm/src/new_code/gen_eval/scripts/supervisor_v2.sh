
#!/bin/bash
#SBATCH --job-name=supervisor_v2
#SBATCH --partition=work_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH -o /projects/0/prjs1589/stonybrook/logs/supervisor_v2-%j.out
#SBATCH -e /projects/0/prjs1589/stonybrook/logs/supervisor_v2-%j.err

#
# Pipeline Supervisor v2 SLURM Job
#
# Enhanced supervisor with:
#   - Runtime GPU/node modification (edit gpu_config.yaml while running)
#   - Priority-based job submission
#   - Improved completion detection
#   - Result aggregation to CSV
#   - Self-backup on start
#
# Usage:
#   1. Edit supervisor_v2_config.yaml with your models/datasets/priorities
#   2. Submit: sbatch supervisor_v2.slurm
#   3. Monitor: cat supervisor_state/dashboard_v2.txt
#   4. Modify GPUs at runtime: edit supervisor_state/gpu_config.yaml
#   5. Stop: scancel <job_id>
#

echo "=========================================="
echo "Pipeline Supervisor v2"
echo "=========================================="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Script directory: $SCRIPT_DIR"
echo "Config file: $SCRIPT_DIR/supervisor_v2_config.yaml"
echo ""

# Load environment
echo "Loading environment..."
cd ~/life-sequencing-dutch/
source requirements/load_venv.sh
cd "$SCRIPT_DIR"

# Check config exists
if [[ ! -f "supervisor_v2_config.yaml" ]]; then
    echo "ERROR: supervisor_v2_config.yaml not found!"
    exit 1
fi

# Create initial GPU config for runtime updates
STATE_DIR="$SCRIPT_DIR/../supervisor_state"
mkdir -p "$STATE_DIR"

if [[ ! -f "$STATE_DIR/gpu_config.yaml" ]]; then
    echo "Creating initial GPU config file..."
    echo "# Runtime GPU Configuration" > "$STATE_DIR/gpu_config.yaml"
    echo "# Edit this file to add/remove GPUs while supervisor is running" >> "$STATE_DIR/gpu_config.yaml"
    echo "# Changes are detected automatically every poll interval" >> "$STATE_DIR/gpu_config.yaml"
    echo "" >> "$STATE_DIR/gpu_config.yaml"
    echo "gpus:" >> "$STATE_DIR/gpu_config.yaml"
    echo "  ossc9424vm1:" >> "$STATE_DIR/gpu_config.yaml"
    echo "    - 0" >> "$STATE_DIR/gpu_config.yaml"
    echo "    - 1" >> "$STATE_DIR/gpu_config.yaml"
    echo "    - 2" >> "$STATE_DIR/gpu_config.yaml"
    echo "    - 3" >> "$STATE_DIR/gpu_config.yaml"
fi

echo ""
echo "Runtime Configuration Files:"
echo "  GPU Config: $STATE_DIR/gpu_config.yaml"
echo "  Dashboard:  $STATE_DIR/dashboard_v2.txt"
echo ""
echo "To modify GPUs at runtime, edit: $STATE_DIR/gpu_config.yaml"
echo ""
echo "=========================================="

# Run supervisor
python supervisor_v2.py --config supervisor_v2_config.yaml

echo ""
echo "=========================================="
echo "Supervisor v2 finished: $(date)"
echo "=========================================="
