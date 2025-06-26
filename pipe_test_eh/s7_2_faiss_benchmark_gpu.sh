#!/bin/bash
#SBATCH --job-name=faiss_benchmark
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=80G
#SBATCH --gpus-per-node=1
#SBATCH -e /home/ehassan/logs/%x.%j.err
#SBATCH -o /home/ehassan/logs/%x.%j.out

source pipe_test_eh/load_sbu_venv.sh

echo "Starting FAISS Benchmark..."
time python -m pipe_test_eh.s7_2_faiss_benchmark_gpu --config=pipe_test_eh/s7_2_faiss_benchmark_gpu_cfg.json
echo "FAISS Benchmark completed."

