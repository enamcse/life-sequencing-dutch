#!/bin/bash
#
#SBATCH --job-name=ft-1
#SBATCH --nodes=1
#SBATCH --nodelist=ossc9424vm1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --mem=100G
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/logs/%j.%x.out

n_gpus=4

echo "job started"
date

source requirements/load_venv.sh
# export CUDA_VISIBLE_DEVICES=0

# ----------- run -----------
cfg="/gpfs/ostor/ossc9424/data/evaluation_sep25/configs-ft-D4-BASE-random/preFer/BASE-random/lr2e-05_bs8_children_post2021.json"

date
export NCCL_SOCKET_IFNAME=ib0
# export NCCL_IB_DISABLE=1

srun python -m pop2vec.llm.src.new_code.finetune_runner "$cfg"

date
echo "job ended successfully"