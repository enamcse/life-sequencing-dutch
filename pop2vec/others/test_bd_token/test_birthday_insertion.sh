#!/bin/bash
#
#SBATCH --job-name=test_birthday_insertion
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH -p thin
#SBATCH -e /projects/0/prjs1589/stonybrook/logs/%x-%j.err
#SBATCH -o /projects/0/prjs1589/stonybrook/logs/%x-%j.out

echo "Job started on $(date)"

cd ~/
source life-sequencing-dutch/requirements/load_venv.sh

echo "Running birthday token insertion test..."
python test_birthday_insertion_verbose.py

echo "Job completed on $(date)"
