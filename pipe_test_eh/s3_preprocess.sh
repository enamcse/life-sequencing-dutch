#!/bin/bash
#
#SBATCH --job-name=preprocess
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=~/life-sequencing-dutch/logs/%x.%j.out
#SBATCH --error=~/life-sequencing-dutch/logs/%x.%j.err
# #SBATCH -e logs/%x.%j.err
# #SBATCH -o logs/%x.%j.out
# #SBATCH --partition=rome
#SBATCH --mem=30G

echo "job started"

date
pwd
source load_sbu_venv.sh
python -m pipe_test_eh.preprocess_data pipe_test_eh/s3_preprocess_cfg.json

echo "job ended successfully"