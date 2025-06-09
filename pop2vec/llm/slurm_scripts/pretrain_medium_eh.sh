#!/bin/bash
#SBATCH --job-name=pretrain_medium
#SBATCH --output=/home/ehassan/pretrain_out.txt
#SBATCH --error=/home/ehassan/pretrain_err.txt
#SBATCH --time=0-12:00:00
#SBATCH --mem=32000
#SBATCH --gres=gpu:1


echo "job started"

#Activate the conda environment
source /miniconda3/etc/profile.d/conda.sh
conda activate pop2vec

# Run the pretraining script
date
time python -m pop2vec.llm.src.new_code.pretrain \
    --config pop2vec/llm/configs/OSSC/pretrain_cfg_medium.json

echo "job ended successfully"