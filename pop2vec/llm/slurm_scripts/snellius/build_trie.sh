#!/bin/bash
#
#SBATCH --job-name=build_trie
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
# #SBATCH --mem=64G
#SBATCH -p gpu_h100
#SBATCH --gpus-per-node=2
#SBATCH -e /projects/0/prjs1589/stonybrook/logs/%x-%j.err
#SBATCH -o /projects/0/prjs1589/stonybrook/logs/%x-%j.out

echo "==========================================
SEQUENCE TRIE BUILDER
=========================================="
echo "Job started on $(date)"
echo "Node: $(hostname)"
echo "=========================================="
echo "RESOURCES:"
echo "  - CPUs allocated: ${SLURM_CPUS_PER_TASK}"
echo "  - Memory allocated: ${SLURM_MEM_PER_NODE}M"
echo "=========================================="

cd ~/life-sequencing-dutch/
source requirements/load_venv.sh

# Configuration file path
CONFIG_FILE="pop2vec/llm/configs/Snellius/build_trie_config.json"

echo "Starting trie building..."
echo "Config file: $CONFIG_FILE"
echo ""

# Run the trie builder script
python -m pop2vec.llm.src.new_code.build_sequence_trie $CONFIG_FILE

echo ""
echo "=========================================="
echo "Trie building completed successfully"
echo "Job ended on $(date)"
echo "=========================================="
