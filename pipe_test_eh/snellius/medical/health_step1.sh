#!/bin/bash
#SBATCH --job-name=health_step1_convert
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.out


echo "⏳ Starting health step1 at $(date) on $(hostname)"

# Activate environment
cd /gpfs/ostor/ossc9424/users/enam/life-sequencing-dutch
source /gpfs/ostor/ossc9424/users/enam/life-sequencing-dutch/requirements/load_venv.sh

# Change to data directory
cd /gpfs/ostor/ossc9424/data/eh2

# Paths (edit if needed)
CFG=configs/health_step1_cfg.json
PY=scripts/health_step1.py

# Run conversion
python $PY --cfg $CFG --stage convert

echo "✅ Finished health step1 at $(date)"
