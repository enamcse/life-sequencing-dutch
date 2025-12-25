#!/usr/bin/env python3
"""
Check Progress of Generative Evaluation Jobs

Monitors completion status, checks for errors, and summarizes progress.

Usage:
    python check_progress.py --experiment exp_n10_c100
    python check_progress.py --all
"""

import argparse
import os
import yaml
from pathlib import Path
from datetime import datetime
import subprocess


def get_slurm_jobs() -> dict:
    """Get running/pending SLURM jobs for current user."""
    try:
        result = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', 'unknown'), '-h', '-o', '%i %j %T %M'],
            capture_output=True, text=True
        )
        jobs = {}
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split()
                if len(parts) >= 4:
                    job_id, name, state, time = parts[0], parts[1], parts[2], parts[3]
                    jobs[name] = {'job_id': job_id, 'state': state, 'time': time}
        return jobs
    except Exception as e:
        print(f"Warning: Could not get SLURM jobs: {e}")
        return {}


def check_experiment_progress(exp_name: str, output_base: Path, slurm_dir: Path):
    """Check progress for a specific experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print('='*60)
    
    # Find manifest
    manifest_path = slurm_dir / f"manifest_{exp_name}.yaml"
    if not manifest_path.exists():
        print(f"  Manifest not found: {manifest_path}")
        return
    
    with open(manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)
    
    scripts = manifest.get('scripts', [])
    running_jobs = get_slurm_jobs()
    
    print(f"\nModels: {len(scripts)}")
    print("-" * 60)
    
    summary = {
        'generation': {'completed': 0, 'running': 0, 'pending': 0, 'failed': 0, 'not_started': 0},
        'statistics': {'completed': 0, 'running': 0, 'pending': 0, 'failed': 0, 'not_started': 0},
    }
    
    for script_info in scripts:
        model = script_info['model']
        output_dir = Path(script_info['output_dir'])
        
        print(f"\n  {model}:")
        
        # Check generation
        gen_job_name = f"gen_{model}_{exp_name}"
        sequences_path = output_dir / "sequences.parquet"
        
        if sequences_path.exists():
            size_mb = sequences_path.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(sequences_path.stat().st_mtime)
            print(f"    Generation: ✓ Complete ({size_mb:.1f} MB, {mtime:%Y-%m-%d %H:%M})")
            summary['generation']['completed'] += 1
        elif gen_job_name in running_jobs:
            job = running_jobs[gen_job_name]
            state = job['state']
            if state == 'RUNNING':
                print(f"    Generation: ⟳ Running (Job {job['job_id']}, {job['time']})")
                summary['generation']['running'] += 1
            else:
                print(f"    Generation: ⏳ Pending (Job {job['job_id']}, {state})")
                summary['generation']['pending'] += 1
        else:
            # Check for error logs
            log_pattern = f"gen_{model}_{exp_name}-*.err"
            log_dir = Path("/projects/0/prjs1589/stonybrook/logs")
            err_logs = list(log_dir.glob(log_pattern)) if log_dir.exists() else []
            
            if err_logs:
                latest_log = max(err_logs, key=lambda p: p.stat().st_mtime)
                if latest_log.stat().st_size > 0:
                    print(f"    Generation: ✗ Failed (see {latest_log})")
                    summary['generation']['failed'] += 1
                else:
                    print(f"    Generation: ? Not started")
                    summary['generation']['not_started'] += 1
            else:
                print(f"    Generation: ? Not started")
                summary['generation']['not_started'] += 1
        
        # Check statistics
        stats_job_name = f"stats_{model}_{exp_name}"
        stats_path = output_dir / "statistics.csv"
        
        if stats_path.exists():
            size_mb = stats_path.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(stats_path.stat().st_mtime)
            print(f"    Statistics:  ✓ Complete ({size_mb:.1f} MB, {mtime:%Y-%m-%d %H:%M})")
            summary['statistics']['completed'] += 1
        elif stats_job_name in running_jobs:
            job = running_jobs[stats_job_name]
            state = job['state']
            if state == 'RUNNING':
                print(f"    Statistics:  ⟳ Running (Job {job['job_id']}, {job['time']})")
                summary['statistics']['running'] += 1
            else:
                print(f"    Statistics:  ⏳ Pending (Job {job['job_id']}, {state})")
                summary['statistics']['pending'] += 1
        else:
            if sequences_path.exists():
                print(f"    Statistics:  ? Ready to run")
            else:
                print(f"    Statistics:  - Waiting for generation")
            summary['statistics']['not_started'] += 1
    
    # Print summary
    print(f"\n{'-'*60}")
    print("Summary:")
    print(f"  Generation:  {summary['generation']['completed']} complete, "
          f"{summary['generation']['running']} running, "
          f"{summary['generation']['pending']} pending, "
          f"{summary['generation']['failed']} failed")
    print(f"  Statistics:  {summary['statistics']['completed']} complete, "
          f"{summary['statistics']['running']} running, "
          f"{summary['statistics']['pending']} pending")
    
    return summary


def check_all_experiments(slurm_dir: Path, output_base: Path):
    """Check progress for all experiments."""
    manifests = list(slurm_dir.glob("manifest_*.yaml"))
    
    if not manifests:
        print("No experiments found")
        return
    
    print(f"Found {len(manifests)} experiments")
    
    for manifest_path in sorted(manifests):
        exp_name = manifest_path.stem.replace('manifest_', '')
        check_experiment_progress(exp_name, output_base, slurm_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Check progress of generative evaluation jobs"
    )
    parser.add_argument("--experiment", "-e", help="Experiment name")
    parser.add_argument("--all", "-a", action="store_true", help="Check all experiments")
    parser.add_argument("--output-dir", help="Output base directory")
    parser.add_argument("--slurm-dir", help="SLURM scripts directory")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    slurm_dir = Path(args.slurm_dir) if args.slurm_dir else script_dir / "slurm_scripts"
    output_base = Path(args.output_dir) if args.output_dir else Path("/projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval")
    
    if args.all:
        check_all_experiments(slurm_dir, output_base)
    elif args.experiment:
        check_experiment_progress(args.experiment, output_base, slurm_dir)
    else:
        parser.error("Either --experiment or --all is required")


if __name__ == "__main__":
    main()
