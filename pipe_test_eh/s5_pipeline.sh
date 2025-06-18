#!/bin/bash
#
#SBATCH --job-name=pipeline
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 18 
#SBATCH --nodes=1
#SBATCH --time=02:30:00
#SBATCH --mem=80G
# #SBATCH -p fat_rome
#SBATCH -e /home/ehassan/logs/%x.%j.err
#SBATCH -o /home/ehassan/logs/%x.%j.out

echo "job started"

date
source pipe_test_eh/load_sbu_venv.sh
srun python -m pipe_test_eh.s5_pipeline pipe_test_eh/s5_pipeline_cfg.json

echo "job ended"