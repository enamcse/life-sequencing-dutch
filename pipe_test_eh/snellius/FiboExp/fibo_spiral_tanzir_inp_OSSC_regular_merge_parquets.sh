#!/bin/bash
#
#SBATCH --job-name=merge-parquet
#SBATCH --time=05:10:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.out
#SBATCH --nodelist=ossc9424vm1

echo "✅ Job started on $(hostname) at $(date)"

# Activate environment
cd /gpfs/ostor/ossc9424/users/tanzir/life-sequencing-dutch
source /gpfs/ostor/ossc9424/users/tanzir/life-sequencing-dutch/requirements/load_venv.sh

# Change to data directory
cd /gpfs/ostor/ossc9424/data/eh2/fibo_tanzir

# Run the merge script
time python fibo_spiral_tanzir_inp_OSSC_merge_parquets.py --input_dir "/gpfs/ostor/ossc9424/data/eh2/fib_spiral_tanzir_regular" --prefix "population_" --output_file "/gpfs/ostor/ossc9424/data/eh2/fib_spiral_tanzir_regular/population_rinpersoon_year_bucketID_regular.parquet"

echo "✅ Job finished at $(date)"
