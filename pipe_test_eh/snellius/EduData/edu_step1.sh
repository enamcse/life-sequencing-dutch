#!/bin/bash
#SBATCH --job-name=edu_step1_convert
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.out

echo "✅ Job started on $(hostname) at $(date)"

# Activate environment
cd /gpfs/ostor/ossc9424/users/enam/life-sequencing-dutch
source /gpfs/ostor/ossc9424/users/enam/life-sequencing-dutch/requirements/load_venv.sh

# Change to data directory
cd /gpfs/ostor/ossc9424/data/eh2

# Paths (edit if needed)
CFG=configs/edu_step1_cfg.json
PY=edu_step1_pipeline.py

# ------------- STAGE 0+1: convert & harmonize -----------------
python $PY --cfg $CFG --stage convert

# ------------- STAGE 2: statistics + correlations -------------
# python $PY --cfg $CFG --stage stats_corr

echo "✅ Job finished at $(date)"
