#!/bin/bash
#
#SBATCH --job-name=ft-twin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --time=120:00:00
#SBATCH --mem=100G
#SBATCH -p gpu_a100
#SBATCH -e /projects/0/prjs1589/stonybrook/logs/%j.%x.err
#SBATCH -o /projects/0/prjs1589/stonybrook/logs/%j.%x.out

n_gpus=1

echo "job started"
date

source requirements/load_venv.sh
# export CUDA_VISIBLE_DEVICES=0

# ----------- run -----------
cfg="pop2vec/llm/src/hparams/snellius/lr2e-05_bs8_is_twin.json"

date
export NCCL_SOCKET_IFNAME=ib0
# export NCCL_IB_DISABLE=1

srun -N1 -n1 --gpus=1 python -m pop2vec.llm.src.new_code.finetune_runner "$cfg"

date
echo "job ended successfully"