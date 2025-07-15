#!/bin/bash
#SBATCH --job-name=bucket-plot
#SBATCH -e /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.out
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -p comp_env

echo "✅ Job started on $(hostname) at $(date)"
# echo "✅ CSV file: $1"
# echo "✅ Year: $2"

# Activate environment
cd /gpfs/ostor/ossc9424/users/tanzir/life-sequencing-dutch
source /gpfs/ostor/ossc9424/users/tanzir/life-sequencing-dutch/requirements/load_venv.sh

# Change to data directory
cd /gpfs/ostor/ossc9424/data/eh2

# Run the plotting script
# python plot_bucket_distribution.py --csv_file "$1" --year "$2"
python plot_bucket_distribution.py --csv_file /gpfs/ostor/ossc9424/data/eh2/fib_spiral_pop_vs_LISS/bucket_summary_net_2009.csv --year 2009


echo "✅ Job finished at $(date)"
