#!/bin/bash
#SBATCH --job-name=network_step1
#SBATCH --time=04:00:00
#SBATCH --mem=320G
#SBATCH --cpus-per-task=4
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.out

echo "⏳ Starting network Step 1 at $(date)"

# Activate environment
cd /gpfs/ostor/ossc9424/users/enam/life-sequencing-dutch
source /gpfs/ostor/ossc9424/users/enam/life-sequencing-dutch/requirements/load_venv.sh

# Change to data directory
cd /gpfs/ostor/ossc9424/data/eh2

# Adjust paths if needed
CFG=configs/network_step1_cfg.json
PY=scripts/network_step1.py

python $PY --cfg $CFG

echo "✅ Finished network Step 1 at $(date)"
