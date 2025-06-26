
#!/bin/bash
#SBATCH --job-name=annoy_benchmark
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH -e /home/ehassan/logs/%x.%j.err
#SBATCH -o /home/ehassan/logs/%x.%j.out

source pipe_test_eh/load_sbu_venv.sh

echo "Starting Annoy Benchmark..."
time python -m pipe_test_eh.s7_annoy_benchmark --config=pipe_test_eh/s7_annoy_benchmark_cfg.json
echo "Annoy Benchmark completed."
