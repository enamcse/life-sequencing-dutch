#!/usr/bin/env python3
"""
Check Progress of Generative Evaluation Jobs

Monitors completion status, checks for errors, and summarizes progress.
Displays results in a table format with experiments as columns and models as rows.

Features:
- Table view with models as rows, experiments as columns
- Status symbols: ✓ (complete), ⟳ (running), ⏳ (pending), ✗ (failed), - (not started)
- Parse log files for job details (start/end time, duration, job ID, status)
- Summary statistics (success/failure counts, average run time)
- Separate views for generation and statistics jobs

Usage:
    python check_progress.py                    # Show table for all experiments
    python check_progress.py --experiment exp_n10_c100  # Single experiment
    python check_progress.py --detailed         # Detailed view with timings
    python check_progress.py --summary          # Summary statistics only
    python check_progress.py --log-dir /path/to/logs    # Custom log directory
"""

import argparse
import os
import re
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import subprocess


# Default paths
DEFAULT_LOG_DIR = "/gpfs/ostor/ossc9424/logs"
DEFAULT_OUTPUT_DIR = "/projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval"

# Status symbols
STATUS_SYMBOLS = {
    'completed': '✓',
    'running': '⟳',
    'pending': '⏳',
    'failed': '✗',
    'not_started': '-',
    'unknown': '?',
}


@dataclass
class JobInfo:
    """Information about a single job."""
    model: str
    experiment: str
    job_type: str  # 'gen' or 'stats'
    status: str  # 'completed', 'running', 'pending', 'failed', 'not_started', 'unknown'
    job_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    output_file: Optional[str] = None
    error_file: Optional[str] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    
    @property
    def symbol(self) -> str:
        return STATUS_SYMBOLS.get(self.status, '?')
    
    @property
    def duration_str(self) -> str:
        if self.duration is None:
            return "-"
        total_seconds = int(self.duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h{minutes}m"
        elif minutes > 0:
            return f"{minutes}m{seconds}s"
        else:
            return f"{seconds}s"


def get_slurm_jobs() -> Dict[str, Dict]:
    """Get running/pending SLURM jobs for current user."""
    try:
        result = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', 'unknown'), '-h', '-o', '%i %j %T %M'],
            capture_output=True, text=True, timeout=30
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


@dataclass
class ExperimentStats:
    """Statistics for a set of jobs."""
    total: int = 0
    completed: int = 0
    failed: int = 0
    running: int = 0
    pending: int = 0
    total_duration_sec: float = 0.0
    durations: List[float] = field(default_factory=list)

    @property
    def avg_duration(self) -> str:
        if not self.durations:
            return "-"
        avg_sec = sum(self.durations) / len(self.durations)
        return str(timedelta(seconds=int(avg_sec)))


def print_time_matrix(job_matrix: Dict[str, Dict[str, JobInfo]]):
    """Print a matrix of execution times (Experiment vs Model)."""
    experiments = sorted(list(set(exp for models in job_matrix.values() for exp in models.keys())))
    models = sorted(list(job_matrix.keys()))
    
    print("\n" + "="*80)
    print("EXECUTION TIME MATRIX (Generation Phase)")
    print("="*80)
    
    # Header
    print(f"{'Model':<30} |", end="")
    for exp in experiments:
        print(f" {exp:<20} |", end="")
    print(f" {'AVG':<10} |")
    print("-" * (30 + 23 * len(experiments) + 13))
    
    # Rows
    for model in models:
        print(f"{model:<30} |", end="")
        model_durations = []
        
        for exp in experiments:
            job = job_matrix.get(model, {}).get(exp)
            duration_str = "-"
            
            if job and job.job_type == 'gen':
                if job.duration:
                    duration_str = job.duration_str
                    model_durations.append(job.duration.total_seconds())
                elif job.status == 'failed':
                    duration_str = "FAILED"
                elif job.status == 'running':
                    duration_str = "RUNNING"
            
            print(f" {duration_str:<20} |", end="")
        
        # Row Average
        if model_durations:
            avg = sum(model_durations) / len(model_durations)
            avg_str = str(timedelta(seconds=int(avg)))
        else:
            avg_str = "-"
        print(f" {avg_str:<10} |")

    print("-" * (30 + 23 * len(experiments) + 13))


def print_experiment_statistics(jobs_list: List[JobInfo]):
    """Print summary statistics for models and experiments."""
    exp_stats = {}
    model_stats = {}
    
    for job in jobs_list:
        if job.job_type != 'gen': continue
        
        # Init stats objects
        if job.experiment not in exp_stats: exp_stats[job.experiment] = ExperimentStats()
        if job.model not in model_stats: model_stats[job.model] = ExperimentStats()
        
        # Update counts
        exp_stats[job.experiment].total += 1
        model_stats[job.model].total += 1
        
        if job.status == 'completed':
            exp_stats[job.experiment].completed += 1
            model_stats[job.model].completed += 1
        elif job.status == 'failed':
            exp_stats[job.experiment].failed += 1
            model_stats[job.model].failed += 1
        
        # Update timings
        if job.duration:
            sec = job.duration.total_seconds()
            exp_stats[job.experiment].durations.append(sec)
            model_stats[job.model].durations.append(sec)

    print("\n" + "="*80)
    print("STATISTICS SUMMARY")
    print("="*80)
    
    print("\nBy Experiment:")
    print(f"{'Experiment':<25} | {'Comp':<5} | {'Fail':<5} | {'Avg Time':<10}")
    print("-" * 55)
    for exp, stats in exp_stats.items():
        print(f"{exp:<25} | {stats.completed:<5} | {stats.failed:<5} | {stats.avg_duration:<10}")
        
    print("\nBy Model:")
    print(f"{'Model':<30} | {'Comp':<5} | {'Fail':<5} | {'Avg Time':<10}")
    print("-" * 60)
    for model, stats in model_stats.items():
        print(f"{model:<30} | {stats.completed:<5} | {stats.failed:<5} | {stats.avg_duration:<10}")
    print()


def parse_datetime(date_str: str) -> Optional[datetime]:
    """Parse datetime from log file format."""
    if not date_str:
        return None
    
    # Try various formats
    formats = [
        "%a %b %d %H:%M:%S %Z %Y",  # "Tue Jan  7 10:30:45 CET 2026"
        "%a %b  %d %H:%M:%S %Z %Y",  # "Tue Jan  7 10:30:45 CET 2026" (double space)
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]
    
    # Also try with timezone stripped
    date_clean = re.sub(r'\s+[A-Z]{3,4}\s+', ' ', date_str)
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            pass
        try:
            return datetime.strptime(date_clean.strip(), fmt.replace(' %Z', ''))
        except ValueError:
            pass
    
    return None


def parse_log_file(log_path: Path) -> Dict:
    """Parse a log file for started/completed times."""
    result = {
        'started': None,
        'completed': None,
        'job_id': None,
        'node': None,
        'error': None,
    }
    
    if not log_path.exists():
        return result
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Look for Started: line
        started_match = re.search(r'Started:\s*(.+)$', content, re.MULTILINE)
        if started_match:
            result['started'] = parse_datetime(started_match.group(1))
        
        # Look for Completed: line
        completed_match = re.search(r'Completed:\s*(.+)$', content, re.MULTILINE)
        if completed_match:
            result['completed'] = parse_datetime(completed_match.group(1))
        
        # Look for Job ID
        job_id_match = re.search(r'Job ID:\s*(\d+)', content)
        if job_id_match:
            result['job_id'] = job_id_match.group(1)
        
        # Look for Node
        node_match = re.search(r'Node:\s*(\S+)', content)
        if node_match:
            result['node'] = node_match.group(1)
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def find_latest_log(log_dir: Path, pattern: str) -> Optional[Path]:
    """Find the latest log file matching a pattern."""
    if not log_dir.exists():
        return None
    
    logs = list(log_dir.glob(pattern))
    if not logs:
        return None
    
    return max(logs, key=lambda p: p.stat().st_mtime)


def get_job_info(
    model: str, 
    experiment: str, 
    job_type: str,
    slurm_jobs: Dict,
    output_base: Path,
    log_dir: Path,
) -> JobInfo:
    """Get information about a specific job."""
    job_name = f"{job_type}_{model}_{experiment}"
    
    # Check output file existence
    run_output_dir = output_base / model / experiment
    if job_type == 'gen':
        output_file = run_output_dir / "generated_sequences.parquet"
    else:
        # Statistics - look for any statistics file
        output_file = run_output_dir / "statistics_summary.csv"
        if not output_file.exists():
            # Try to find any stats file
            stats_files = list(run_output_dir.glob("statistics_*_summary.csv"))
            if stats_files:
                output_file = stats_files[0]
    
    # Find log files
    out_log_pattern = f"{job_name}-*.out"
    err_log_pattern = f"{job_name}-*.err"
    
    latest_out = find_latest_log(log_dir, out_log_pattern)
    latest_err = find_latest_log(log_dir, err_log_pattern)
    
    # Parse log file for timing info
    log_info = {}
    if latest_out:
        log_info = parse_log_file(latest_out)
    
    # Determine status
    error_message = None
    
    if output_file.exists():
        status = 'completed'
        job_id = log_info.get('job_id')
        start_time = log_info.get('started')
        end_time = log_info.get('completed')
        duration = None
        if start_time and end_time:
            duration = end_time - start_time
    elif job_name in slurm_jobs:
        job = slurm_jobs[job_name]
        if job['state'] == 'RUNNING':
            status = 'running'
        else:
            status = 'pending'
        job_id = job['job_id']
        start_time = log_info.get('started')
        end_time = None
        duration = None
    else:
        # Check if there's an error log
        if latest_err and latest_err.exists() and latest_err.stat().st_size > 0:
            # Check if it's a real error or just empty stderr
            try:
                with open(latest_err, 'r') as f:
                    err_content = f.read().strip()
                if err_content:
                    error_message = err_content[:200]  # First 200 chars
                    status = 'failed'
                else:
                    status = 'not_started'
            except:
                status = 'failed'
        else:
            status = 'not_started'
        
        job_id = log_info.get('job_id')
        start_time = log_info.get('started')
        end_time = log_info.get('completed')
        duration = None
        if start_time and end_time:
            duration = end_time - start_time
    
    return JobInfo(
        model=model,
        experiment=experiment,
        job_type=job_type,
        status=status,
        job_id=job_id,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        output_file=str(latest_out) if latest_out else None,
        error_file=str(latest_err) if latest_err else None,
        output_path=str(output_file) if output_file.exists() else None,
        error_message=error_message,
    )


def collect_all_jobs(
    models: List[str],
    experiments: List[str],
    output_base: Path,
    log_dir: Path,
) -> Dict[str, Dict[str, Dict[str, JobInfo]]]:
    """Collect job information for all model/experiment combinations.
    
    Returns:
        Dict[job_type][model][experiment] = JobInfo
    """
    slurm_jobs = get_slurm_jobs()
    
    result = {'gen': {}, 'stats': {}}
    
    for model in models:
        result['gen'][model] = {}
        result['stats'][model] = {}
        
        for experiment in experiments:
            result['gen'][model][experiment] = get_job_info(
                model, experiment, 'gen', slurm_jobs, output_base, log_dir
            )
            result['stats'][model][experiment] = get_job_info(
                model, experiment, 'stats', slurm_jobs, output_base, log_dir
            )
    
    return result


def print_table(
    jobs: Dict[str, Dict[str, Dict[str, JobInfo]]],
    models: List[str],
    experiments: List[str],
    job_type: str,
    show_details: bool = False,
):
    """Print a table with models as rows and experiments as columns."""
    title = "Generation" if job_type == 'gen' else "Statistics"
    print(f"\n{'='*60}")
    print(f" {title} Jobs")
    print('='*60)
    
    # Calculate column widths
    model_width = max(len(m) for m in models) + 2
    if show_details:
        exp_widths = [max(len(e), 12) + 2 for e in experiments]
    else:
        exp_widths = [max(len(e), 6) + 2 for e in experiments]
    
    # Header
    header = f"{'Model':<{model_width}}"
    for i, exp in enumerate(experiments):
        # Truncate long experiment names
        exp_display = exp[:exp_widths[i]-2] if len(exp) > exp_widths[i]-2 else exp
        header += f" {exp_display:^{exp_widths[i]}}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for model in models:
        row = f"{model:<{model_width}}"
        for i, exp in enumerate(experiments):
            job = jobs[job_type][model][exp]
            if show_details and job.duration:
                cell = f"{job.symbol} {job.duration_str}"
            else:
                cell = job.symbol
            row += f" {cell:^{exp_widths[i]}}"
        print(row)
    
    # Legend
    print()
    print("Legend: ✓=completed, ⟳=running, ⏳=pending, ✗=failed, -=not started")


def print_summary(
    jobs: Dict[str, Dict[str, Dict[str, JobInfo]]],
    models: List[str],
    experiments: List[str],
):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(" Summary Statistics")
    print('='*60)
    
    for job_type in ['gen', 'stats']:
        title = "Generation" if job_type == 'gen' else "Statistics"
        
        total = 0
        completed = 0
        running = 0
        pending = 0
        failed = 0
        not_started = 0
        
        durations = []
        
        for model in models:
            for exp in experiments:
                job = jobs[job_type][model][exp]
                total += 1
                if job.status == 'completed':
                    completed += 1
                    if job.duration:
                        durations.append(job.duration.total_seconds())
                elif job.status == 'running':
                    running += 1
                elif job.status == 'pending':
                    pending += 1
                elif job.status == 'failed':
                    failed += 1
                elif job.status == 'not_started':
                    not_started += 1
        
        print(f"\n{title}:")
        print(f"  Total: {total}")
        print(f"  Completed: {completed} ({completed/total*100:.1f}%)")
        print(f"  Running: {running}")
        print(f"  Pending: {pending}")
        print(f"  Failed: {failed}")
        print(f"  Not Started: {not_started}")
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            avg_str = str(timedelta(seconds=int(avg_duration)))
            print(f"  Avg Duration: {avg_str}")


def main():
    parser = argparse.ArgumentParser(description="Check progress of generative evaluation jobs")
    parser.add_argument("--config", help="Path to run config YAML (finds models/exp)")
    parser.add_argument("--experiment", "-e", nargs="+", help="Specific experiment(s) to check")
    parser.add_argument("--models", "-m", nargs="+", help="Specific model(s) to check")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="SLURM log directory")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output base directory")
    parser.add_argument("--detailed", action="store_true", help="Show detailed output")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument("--stats", action="store_true", help="Show detailed statistics (time matrix)")
    args = parser.parse_args()
    
    # Auto-detect models and experiments if not provided
    models = args.models
    experiments = args.experiment
    
    if not models or not experiments:
        # Try to find all directories in output_dir
        output_base = Path(args.output_dir)
        if output_base.exists():
            if not models:
                models = [d.name for d in output_base.iterdir() if d.is_dir()]
                models.sort()
            
            if not experiments and models:
                # Find experiments common to all models? Or just union?
                # Let's find union of all experiments found in first model
                first_model_dir = output_base / models[0]
                if first_model_dir.exists():
                    experiments = [d.name for d in first_model_dir.iterdir() if d.is_dir()]
                    experiments.sort()
    
    # Fallback default values
    if not models:
        models = ["model_v1_gen_20251117"]
    if not experiments:
        experiments = ["exp_n10_c100_h20_g100"]
    
    print(f"Checking progress for {len(models)} models and {len(experiments)} experiments...")
    
    log_dir = Path(args.log_dir)
    output_base = Path(args.output_dir)
    
    # Collect all job info
    jobs_map = collect_job_info(models, experiments, output_base, log_dir)
    
    if not args.summary and not args.stats:
        print_table(jobs_map, models, experiments, 'gen', args.detailed)
        print_table(jobs_map, models, experiments, 'stats', args.detailed)
    
    if args.summary or args.stats or args.detailed:
        print_summary(jobs_map, models, experiments)
    
    if args.stats:
        # Convert map to list for stats
        all_jobs = []
        for stage in ['gen', 'stats']:
            for m in models:
                for e in experiments:
                    all_jobs.append(jobs_map[stage][m][e])
        
        print_experiment_statistics(all_jobs)
        print_time_matrix(jobs_map['gen'])


if __name__ == "__main__":
    main()
