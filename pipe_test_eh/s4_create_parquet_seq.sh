#!/bin/bash
#
#SBATCH --job-name=create_parquet_seq
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --mem=30G
# #SBATCH -p rome 
#SBATCH -e /home/ehassan/logs/%x.%j-s4.err
#SBATCH -o /home/ehassan/logs/%x.%j-s4.out

echo "job started"


date
source pipe_test_eh/load_sbu_venv.sh
time python -m pipe_test_eh.s4_create_life_seq_parquets pipe_test_eh/s4_create_parquet_seq_cfg.json

echo "job ended"
