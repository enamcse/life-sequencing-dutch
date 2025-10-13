#!/bin/bash
#
#SBATCH --job-name=pretrain-D4-medium-event
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=ossc9424vm1
#SBATCH --time=120:00:00
#SBATCH --mem=100G
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/homedir/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/homedir/logs/%j.%x.out

n_gpus=4

echo "job started"
date

source requirements/load_venv.sh

# export CUDA_VISIBLE_DEVICES=0

HP="pop2vec/llm/src/hparams/pretrain-D4-medium-event.txt"

date
export NCCL_SOCKET_IFNAME=ib0
# export NCCL_IB_DISABLE=1

srun python -m pop2vec.llm.src.new_code.pretrain --devices $n_gpus --hparams $HP

date
echo "job ended successfully"