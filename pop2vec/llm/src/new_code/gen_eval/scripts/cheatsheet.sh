#!/bin/bash
# ============================================================================
# QUICK REFERENCE - Pipeline Supervisor
# ============================================================================
# Copy this to your secured server and run from the scripts/ directory
# ============================================================================

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════════════╗
║                    PIPELINE SUPERVISOR - CHEAT SHEET                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  STEP 1: PREPARE (one time)                                                  ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  # Generate SLURM scripts for all models × experiments                       ║
║  python generate_slurm.py --config ../config/experiments_config.yaml \       ║
║      --gpus "ossc9424vm1:0,1,2,3"                                            ║
║                                                                              ║
║  # Edit supervisor config                                                    ║
║  vim supervisor_config.yaml                                                  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  STEP 2: RUN                                                                 ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  sbatch supervisor.slurm                                                     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  STEP 3: MONITOR                                                             ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  # View dashboard (updates every 60s)                                        ║
║  cat ../supervisor_state/dashboard.txt                                       ║
║                                                                              ║
║  # Watch continuously                                                        ║
║  watch -n 30 cat ../supervisor_state/dashboard.txt                           ║
║                                                                              ║
║  # Check SLURM queue                                                         ║
║  squeue -u $USER                                                             ║
║                                                                              ║
║  # Check supervisor log                                                      ║
║  tail -f /projects/0/prjs1589/stonybrook/logs/supervisor-*.out               ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  STEP 4: STOP (when needed)                                                  ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  # Find supervisor job ID                                                    ║
║  squeue -u $USER | grep supervisor                                           ║
║                                                                              ║
║  # Cancel it                                                                 ║
║  scancel <job_id>                                                            ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  TROUBLESHOOTING                                                             ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  # Check specific job logs                                                   ║
║  ls /projects/0/prjs1589/stonybrook/logs/gen_*.out                           ║
║  cat /projects/0/prjs1589/stonybrook/logs/gen_model_v1_*-*.err               ║
║                                                                              ║
║  # Reset everything                                                          ║
║  scancel -u $USER                                                            ║
║  rm -rf ../supervisor_state/                                                 ║
║  sbatch supervisor.slurm                                                     ║
║                                                                              ║
║  # Manual job submission (if needed)                                         ║
║  sbatch ../slurm_scripts/gen_model_v1_exp_n10_c100_h20_g100.sh               ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DASHBOARD LEGEND                                                            ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  ✓ = Completed successfully                                                  ║
║  ⟳ = Currently running                                                       ║
║  ⏳ = Queued (waiting for resources or dependencies)                          ║
║  ✗ = Failed (check logs)                                                     ║
║  - = Not started yet                                                         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FILE LOCATIONS                                                              ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  Config:     scripts/supervisor_config.yaml                                  ║
║  Dashboard:  supervisor_state/dashboard.txt                                  ║
║  State:      supervisor_state/pipeline_state.json                            ║
║  SLURM logs: /projects/0/prjs1589/stonybrook/logs/                           ║
║  Outputs:    /projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

EOF
