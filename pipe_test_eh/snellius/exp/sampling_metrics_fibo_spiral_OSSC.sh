#!/bin/bash
#
#SBATCH --job-name=fibo-spiral
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:30:00
#SBATCH --mem=128G
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.out

echo "✅ SLURM job started on $(hostname)"
echo "✅ Start time: $(date)"
echo "✅ Job ID: $SLURM_JOB_ID"
echo "✅ Array Task ID: ${SLURM_ARRAY_TASK_ID:-None}"


# Activate environment
cd /gpfs/ostor/ossc9424/users/tanzir/life-sequencing-dutch
source /gpfs/ostor/ossc9424/users/tanzir/life-sequencing-dutch/requirements/load_venv.sh

# Change to data directory
cd /gpfs/ostor/ossc9424/data/eh2

# Determine array index argument
INDEX_ARG=""
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    INDEX_ARG="--index $SLURM_ARRAY_TASK_ID"
fi

echo "✅ Running Python module with INDEX_ARG: '$INDEX_ARG'"


# Run your main Python script
time python -m sampling_metrics_fibo_spiral_OSSC sampling_metrics_fibo_spiral_OSSC_cfg.json $INDEX_ARG

echo "✅ Job ended at $(date)"
