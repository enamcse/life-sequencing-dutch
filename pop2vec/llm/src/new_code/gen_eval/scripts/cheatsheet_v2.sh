#!/bin/bash
#
# Supervisor v2 Quick Reference / Cheat Sheet
# ============================================
#
# Run this script with no arguments to see the cheat sheet:
#   bash cheatsheet_v2.sh
#
# Or run with a command:
#   bash cheatsheet_v2.sh submit
#   bash cheatsheet_v2.sh status
#   bash cheatsheet_v2.sh add-gpu
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/../supervisor_state"
EXPORT_DIR="/projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/exports"
LOG_DIR="/projects/0/prjs1589/stonybrook/logs"

show_help() {
    cat << 'EOF'
==========================================================================
                    SUPERVISOR V2 QUICK REFERENCE
==========================================================================

SUBMIT & MONITOR
----------------
  sbatch supervisor_v2.slurm           # Start supervisor
  cat ../supervisor_state/dashboard_v2.txt   # View dashboard
  tail -f $LOG_DIR/supervisor_v2-*.out       # Watch live logs
  squeue -u $USER                            # Check SLURM queue

STOP SUPERVISOR
---------------
  scancel $(squeue -u $USER -n supervisor_v2 -h -o %i)

RUNTIME GPU CONFIGURATION
-------------------------
  # Add/remove GPUs while supervisor is running
  vim ../supervisor_state/gpu_config.yaml
  
  # Example gpu_config.yaml:
  # gpus:
  #   ossc9424vm1: [0, 1, 2, 3]
  #   ossc9424vm2: [0, 1]        # Add/remove nodes here

PRIORITY LEVELS (set max_priority_level in config)
--------------------------------------------------
  Level 1:  Base (t=0.8, k=20, h=20, g=100)     →  25 experiments
  Level 2:  +k=1                                 →  50 experiments
  Level 3:  +k=v (vocab size)                    →  75 experiments
  Level 4:  +t=0.1                               → 150 experiments
  Level 5:  +t=1.0                               → 225 experiments
  Level 6:  +k=10                                → 300 experiments
  Level 7:  +t=0.3                               → 400 experiments
  Level 8:  +k=5                                 → 500 experiments
  Level 9:  +h=10                                → 1000 experiments
  Level 10: +g=50                                → 2000 experiments

EXPERIMENT NAMING
-----------------
  Format: exp_n{n}_c{c}_h{h}_g{g}_k{k}_t{temp}_{model}_{dataset}
  Example: exp_n100_c100_h20_g100_k20_t08_GenBASE_GD0

DATASETS
--------
  GD0: Childhood/Young (age 1-30 at pos 1000)
  GD1: Middle-age (30-49)
  GD2: Late middle-age (50-69)
  GD3: Old age (70-99)
  GD4: Mixed (20% 0-29, 25% 30-49, 25% 50-69, 20% 70-99, 10% death)
  GDB0-GDB4: Birthday-token versions (same indices)

MODELS
------
  Gen-medium
  Gen-BASE
  Gen-medium-bd
  Gen-BASE-bd
  Gen-BASE-bd-partial

EXPORT FILES
------------
  statistics_block_summary.csv   - Block-wise comparisons (12 types)
  statistics_decade_summary.csv  - Decade-wise token frequencies
  token_counts_merged.csv        - Token counts with metadata

CHECK EXPORTS
-------------
  ls -la /projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/exports/

CREATE DATASETS
---------------
  python create_generative_datasets.py --config ../config/generative_datasets_config.yaml

COMPLETION MARKERS (edit supervisor_v2.py to add more)
-----------------------------------------------------
  Generation: "Generation Complete!"
  Statistics: "Statistics Complete!" or "All Steps Completed:"
  
==========================================================================
EOF
}

case "$1" in
    submit|start)
        echo "Submitting supervisor v2..."
        cd "$SCRIPT_DIR"
        sbatch supervisor_v2.slurm
        ;;
    status|dashboard)
        cat "$STATE_DIR/dashboard_v2.txt"
        ;;
    logs)
        echo "Tailing supervisor logs..."
        tail -f "$LOG_DIR"/supervisor_v2-*.out 2>/dev/null || echo "No logs found"
        ;;
    queue)
        squeue -u "$USER"
        ;;
    stop)
        echo "Stopping supervisor..."
        scancel $(squeue -u "$USER" -n supervisor_v2 -h -o %i 2>/dev/null)
        ;;
    exports)
        ls -la "$EXPORT_DIR"
        ;;
    add-gpu)
        echo "Edit GPU config:"
        echo "  vim $STATE_DIR/gpu_config.yaml"
        echo ""
        echo "Current config:"
        cat "$STATE_DIR/gpu_config.yaml" 2>/dev/null || echo "No config yet"
        ;;
    state)
        cat "$STATE_DIR/pipeline_state_v2.json"
        ;;
    *)
        show_help
        ;;
esac
