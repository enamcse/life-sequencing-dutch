#!/bin/bash
#SBATCH --job-name=eval-buckets-regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:30:00
#SBATCH --mem=128G
#SBATCH -p comp_env
#SBATCH --nodelist=ossc9424vm1
#SBATCH -e /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.out

echo "✅ Job started on $(hostname) at $(date)"

# Activate environment
cd /gpfs/ostor/ossc9424/users/tanzir/life-sequencing-dutch
source /gpfs/ostor/ossc9424/users/tanzir/life-sequencing-dutch/requirements/load_venv.sh

# Change to data directory
cd /gpfs/ostor/ossc9424/data/eh2/fibo_tanzir

# Call Python script with all arguments
python evaluate_bucket_similarity_over_years_OSSC.py \
  --parquet_file "/gpfs/ostor/ossc9424/data/eh2/population_rinpersoon_year_bucketID_regular.parquet" \
  --year1_list "2009 2010" \
  --end_year "2020" \
  --sample_size "1000000" \
  --output_file "/gpfs/ostor/ossc9424/data/eh2/fibo_tanzir/eval_results_regular.csv"

echo "✅ Job finished at $(date)"
