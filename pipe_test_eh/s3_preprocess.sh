#!/bin/bash
#
#SBATCH --job-name=preprocess
#SBATCH -e /home/ehassan/logs/%x.%j.err
#SBATCH -o /home/ehassan/logs/%x.%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=3:00:00
# #SBATCH --partition=rome
#SBATCH --mem=32G

echo "job started"

date
pwd
source pipe_test_eh/load_sbu_venv.sh
python -m pipe_test_eh.s3_preprocess_data pipe_test_eh/s3_preprocess_cfg.json

echo "job ended successfully"