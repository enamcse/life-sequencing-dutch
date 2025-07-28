#!/bin/bash
#SBATCH --job-name=add_net_cluster
#SBATCH --time=04:00:00
#SBATCH --mem=320G
#SBATCH --cpus-per-task=4
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.out

echo "⏳ Starting add_net_cluster at $(date)"

# Activate environment
cd /gpfs/ostor/ossc9424/users/enam/life-sequencing-dutch
source /gpfs/ostor/ossc9424/users/enam/life-sequencing-dutch/requirements/load_venv.sh

# Change to data directory
cd /gpfs/ostor/ossc9424/data/eh2

# Adjust paths if needed
CFG=configs/add_cluster_cfg.json
PY=scripts/add_cluster.py

python $PY --cfg $CFG

echo "✅ Finished add_net_cluster at $(date)"

