#!/bin/bash
#SBATCH --job-name=pretrain_medium
#SBATCH -o /home/ehassan/logs/pretrain_%x.%j.out
#SBATCH -e /home/ehassan/logs/pretrain_%x.%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
# #SBATCH -p comp_env       # Not sure if there is any partition defined by the admin
#SBATCH --time=0-12:00:00
#SBATCH --mem=32G
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