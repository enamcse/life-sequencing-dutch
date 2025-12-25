#!/bin/bash
#
#SBATCH --job-name=fix_birthday_h5
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH -p thin
#SBATCH -e /projects/0/prjs1589/stonybrook/logs/%x-%j.err
#SBATCH -o /projects/0/prjs1589/stonybrook/logs/%x-%j.out

echo "Job started on $(date)"
date

cd ~/life-sequencing-dutch/
source requirements/load_venv.sh

# Configuration file path
CONFIG_FILE="pop2vec/llm/configs/Snellius/fix_birthday_h5_config.json"

echo "Fixing birthday token HDF5 output..."
echo "Config file: $CONFIG_FILE"

# Run the fix script
python -m pop2vec.llm.src.new_code.fix_birthday_h5_output $CONFIG_FILE

date
echo "Fix completed successfully"
