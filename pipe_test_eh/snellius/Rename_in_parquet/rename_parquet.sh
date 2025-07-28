#!/bin/bash
#SBATCH --job-name=rename_parquet
#SBATCH --time=10:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.out

CFG=configs/rename_parquet_cfg.json

# Activate environment
cd /gpfs/ostor/ossc9424/users/enam/life-sequencing-dutch
source /gpfs/ostor/ossc9424/users/enam/life-sequencing-dutch/requirements/load_venv.sh

# Change to data directory
cd /gpfs/ostor/ossc9424/data/eh2

echo "⏳ Running rename_parquet with $CFG"
python scripts/rename_parquet.py --cfg $CFG

echo "✅ Renaming finished at $(date)"
