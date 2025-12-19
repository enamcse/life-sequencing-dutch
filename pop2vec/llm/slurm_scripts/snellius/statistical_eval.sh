#!/bin/bash
#SBATCH --job-name=statistical_eval
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH -p gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH -e /projects/0/prjs1589/stonybrook/logs/%x-%j.err
#SBATCH -o /projects/0/prjs1589/stonybrook/logs/%x-%j.out

echo "Statistical Evaluation Job started on $(date)"
date

cd ~/life-sequencing-dutch/
source requirements/load_venv.sh

HP="pop2vec/llm/src/hparams/snellius/statistical_eval_hparams.txt"

echo "Running statistical evaluation with config: $HP"
date

python ~/life-sequencing-dutch/pop2vec/llm/scripts/statistical_evaluation.py \
    --hparams $HP

date
echo "Statistical evaluation job ended successfully"
