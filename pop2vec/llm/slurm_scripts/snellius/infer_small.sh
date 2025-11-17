#!/bin/bash
#
#SBATCH --job-name=infer_small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=5:30:00
#SBATCH --mem=100G
#SBATCH -p gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH -e /projects/0/prjs1589/stonybrook/logs/%x-%j.err
#SBATCH -o /projects/0/prjs1589/stonybrook/logs/%x-%j.out

#declare PREFIX="/gpfs/ostor/ossc9424/homedir/"

#export CUDA_VISIBLE_DEVICES=0

n_gpus=1

echo "job started"
date

source requirements/load_venv.sh

cfg="pop2vec/llm/configs/Snellius/infer_cfg_small.json"

date
srun python -m pop2vec.llm.src.new_code.infer_embedding $cfg

date
echo "job ended successfully"
