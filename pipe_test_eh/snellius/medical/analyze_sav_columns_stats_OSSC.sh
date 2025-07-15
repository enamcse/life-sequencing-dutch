#!/bin/bash
#SBATCH --job-name=sav-stats
#SBATCH --time=23:30:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=2
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.out


echo "✅ Job started on $(hostname) at $(date)"
# echo "✅ SAV directory: $1"

# Activate environment
cd /gpfs/ostor/ossc9424/users/tanzir/life-sequencing-dutch
source /gpfs/ostor/ossc9424/users/tanzir/life-sequencing-dutch/requirements/load_venv.sh

# Change to data directory
cd /gpfs/ostor/ossc9424/data/eh2

# Call analysis
python analyze_sav_column_stats_OSSC.py --input_dir "/gpfs/ostor/ossc9424/data/eh2/health_raw" --output_dir "/gpfs/ostor/ossc9424/data/eh2/health_out"

echo "✅ Job finished at $(date)"
