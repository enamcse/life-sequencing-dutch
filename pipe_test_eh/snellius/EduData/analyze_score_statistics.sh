#!/bin/bash
#SBATCH --job-name=score_stats
#SBATCH --time=23:30:00
#SBATCH --mem=28G
#SBATCH --cpus-per-task=2
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.out

# Activate environment
cd /gpfs/ostor/ossc9424/users/tanzir/life-sequencing-dutch
source /gpfs/ostor/ossc9424/users/tanzir/life-sequencing-dutch/requirements/load_venv.sh

# Change to data directory
cd /gpfs/ostor/ossc9424/data/eh2/edu_data

python analyze_score_statistics.py \
    --cito_file='/gpfs/ostor/ossc9424/data/eh2/edu_data/cito_scores.parquet' \
    --ce_file='/gpfs/ostor/ossc9424/data/eh2/edu_data/ce_scores.parquet' \
    --output_dir='/gpfs/ostor/ossc9424/data/eh2/edu_data'

