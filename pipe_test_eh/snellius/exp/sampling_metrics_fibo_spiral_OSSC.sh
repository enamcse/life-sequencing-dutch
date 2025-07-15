#!/bin/bash
#
#SBATCH --job-name=fibo-spiral
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=02:30:00
#SBATCH --mem=300G
#SBATCH -p comp_env 
#SBATCH -e /gpfs/ostor/ossc9424/data/eh/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/data/eh/logs/%j.%x.out

echo "job started"

date
cd /gpfs/ostor/ossc9424/users/tanzir/life-sequencing-dutch
source /gpfs/ostor/ossc9424/users/tanzir/life-sequencing-dutch/requirements/load_venv.sh
cd /gpfs/ostor/ossc9424/data/eh
time python -m sampling_metrics_fibo_spiral_OSSC sampling_metrics_fibo_spiral_OSSC_cfg.json

echo "job ended"
