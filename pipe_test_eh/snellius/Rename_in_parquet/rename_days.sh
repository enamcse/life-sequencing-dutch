#!/bin/bash
#SBATCH --job-name=rename_days
#SBATCH --time=10:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.out

# Usage: sbatch run_rename_days.sh /path/to/parquets /path/to/meta_parquets

DATA_DIR=$1
META_DIR=$2

if [ -z "$DATA_DIR" ] || [ -z "$META_DIR" ]; then
  echo "Usage: $0 <data_dir> <meta_dir>"
  exit 1
fi

echo "⏳ Start Renaming daysSinceFirst → daysSinceFirstEvent at $(date) on $(hostname)"
echo "    data dir: $DATA_DIR"
echo "    meta dir: $META_DIR"

# Activate environment
cd /gpfs/ostor/ossc9424/users/enam/life-sequencing-dutch
source /gpfs/ostor/ossc9424/users/enam/life-sequencing-dutch/requirements/load_venv.sh

# Change to data directory
cd /gpfs/ostor/ossc9424/data/eh2

python rename_days.py --data-dir "$DATA_DIR" --meta-dir "$META_DIR"

DATA_DIR=$1
META_DIR=$2

echo "    data dir: $DATA_DIR"
echo "    meta dir: $META_DIR"

python rename_days.py --data-dir "$DATA_DIR" --meta-dir "$META_DIR"

echo "✅ Renaming finished at $(date)"
