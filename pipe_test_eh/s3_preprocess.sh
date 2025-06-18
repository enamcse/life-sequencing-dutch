#!/bin/bash
#
#SBATCH --job-name=preprocess
#SBATCH -e /home/ehassan/logs/%x.%j.err
#SBATCH -o /home/ehassan/logs/%x.%j.out
# #SBATCH --output=/home/ehassan/logs/%x.%j.out
# #SBATCH --error=/home/ehassan/logs/%x.%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
# #SBATCH --partition=rome
#SBATCH --mem=32G

echo "job started"

date
pwd
source load_sbu_venv.sh
python -m pipe_test_eh.preprocess_data pipe_test_eh/s3_preprocess_cfg.json

echo "job ended successfully"