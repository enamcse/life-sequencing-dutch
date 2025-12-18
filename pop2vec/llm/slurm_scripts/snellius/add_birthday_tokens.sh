#!/bin/bash
#
#SBATCH --job-name=add_birthday_tokens
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH -p gpu_h100
#SBATCH --gpus-per-node=2
#SBATCH -e /projects/0/prjs1589/stonybrook/logs/%x-%j.err
#SBATCH -o /projects/0/prjs1589/stonybrook/logs/%x-%j.out

echo "Job started on $(date)"
echo "Using ${SLURM_CPUS_PER_TASK} CPU cores for parallel processing"
date

cd ~/life-sequencing-dutch/
source requirements/load_venv.sh

# Configuration file path
CONFIG_FILE="pop2vec/llm/configs/Snellius/add_birthdays_config.json"

echo "Starting birthday token preprocessing..."
echo "Config file: $CONFIG_FILE"

# Run the preprocessing script
python -m pop2vec.llm.src.new_code.add_birthday_token_to_preprocess_data $CONFIG_FILE

date
echo "Birthday token preprocessing completed successfully"
