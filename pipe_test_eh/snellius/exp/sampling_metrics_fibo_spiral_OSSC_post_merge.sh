#!/bin/bash
#
#SBATCH --job-name=merge-metrics
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH -e /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/data/eh2/logs/%j.%x.out

echo "✅ Merge job started at $(date)"
cd /gpfs/ostor/ossc9424/data/eh2/fib_spiral_pop_vs_LISS

# Merge CSVs
echo "✅ Merging partial files..."
cat metrics_summary_part_*.csv | head -n 1 > metrics_summary.csv
for f in metrics_summary_part_*.csv; do
    tail -n +2 "$f" >> metrics_summary.csv
done
echo "✅ Merged metrics_summary.csv created."

# Cleanup
echo "✅ Deleting partial files..."
rm -f metrics_summary_part_*.csv

echo "✅ Merge job complete at $(date)"
