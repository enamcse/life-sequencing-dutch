#!/bin/bash
#
#SBATCH --job-name=pretrain_small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --mem=80G
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -e /home/ehassan/logs/%x.%j.err
#SBATCH -o /home/ehassan/logs/%x.%j.out

source pipe_test_eh/load_sbu_venv.sh

#export CUDA_VISIBLE_DEVICES=0

date
time python -m pipe_test_eh.s6_pretrain --config=pipe_test_eh/s6_pretrain_cfg_small.json

echo "job ended successfully"