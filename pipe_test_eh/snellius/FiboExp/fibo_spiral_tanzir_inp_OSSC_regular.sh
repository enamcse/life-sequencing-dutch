#!/bin/bash
#
#SBATCH --job-name=fibo-regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:30:00
#SBATCH --mem=128G
#SBATCH -p comp_env
#SBATCH --nodelist=ossc9424vm1
## #SBATCH --array=0-5
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
cd /gpfs/ostor/ossc9424/data/eh2/fibo_tanzir

# Determine array index argument
INDEX_ARG=""
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    INDEX_ARG="--index $SLURM_ARRAY_TASK_ID"
fi

echo "✅ Running Python module with INDEX_ARG: '$INDEX_ARG'"


# Run your main Python script
time python -m fibo_spiral_tanzir_inp_OSSC fibo_spiral_tanzir_inp_OSSC_regular_cfg.json $INDEX_ARG

echo "✅ Job ended at $(date)"
