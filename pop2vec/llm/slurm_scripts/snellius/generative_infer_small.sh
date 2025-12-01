#!/bin/bash
#
#SBATCH --job-name=generative_infer_small
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=15
#SBATCH --time=100:00:00
#SBATCH --mem=80G
#SBATCH -p gpu_a100
#SBATCH --gpus-per-node=2
#SBATCH -e /projects/0/prjs1589/stonybrook/logs/%x-%j.err
#SBATCH -o /projects/0/prjs1589/stonybrook/logs/%x-%j.out

n_gpus=2

echo "Job started on $(date)"
date

cd ~/life-sequencing-dutch/
source requirements/load_venv.sh

#export CUDA_VISIBLE_DEVICES=0

HP="pop2vec/llm/src/hparams/snellius/regular_generative_hparams_small.txt"

date
srun python -m pop2vec.llm.src.new_code.generative_infer --hparams $HP

date
echo "job ended successfully"
