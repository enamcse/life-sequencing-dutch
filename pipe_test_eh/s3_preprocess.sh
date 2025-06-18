#!/bin/bash
#
#SBATCH --job-name=preprocess
#SBATCH -e /home/ehassan/logs/%x.%j.err
#SBATCH -o /home/ehassan/logs/%x.%j.out
# #SBATCH --output=/home/ehassan/preprocess_out.txt
# #SBATCH --error=/home/ehassan/preprocess_err.txt
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
# #SBATCH --output=~/life-sequencing-dutch/logs/%x.%j.out
# #SBATCH --error=~/life-sequencing-dutch/logs/%x.%j.err
# #SBATCH --partition=rome
#SBATCH --mem=32G

echo "job started"

date
pwd
source load_sbu_venv.sh
python -m pipe_test_eh.preprocess_data pipe_test_eh/s3_preprocess_cfg.json

echo "job ended successfully"