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
                else:
                    not_started += 1
        
        print(f"\n{title}:")
        print(f"  Total jobs:     {total}")
        if total > 0:
            print(f"  Completed:      {completed} ({100*completed/total:.1f}%)")
        else:
            print(f"  Completed:      0")
        print(f"  Running:        {running}")
        print(f"  Pending:        {pending}")
        print(f"  Failed:         {failed}")
        print(f"  Not started:    {not_started}")
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            
            print(f"\n  Timing (completed jobs):")
            print(f"    Average:      {timedelta(seconds=int(avg_duration))}")
            print(f"    Min:          {timedelta(seconds=int(min_duration))}")
            print(f"    Max:          {timedelta(seconds=int(max_duration))}")


def print_detailed_table(
    jobs: Dict[str, Dict[str, Dict[str, JobInfo]]],
    models: List[str],
    experiments: List[str],
):
    """Print detailed information for each job."""
    print(f"\n{'='*60}")
    print(" Detailed Job Information")
    print('='*60)
    
    for job_type in ['gen', 'stats']:
        title = "Generation" if job_type == 'gen' else "Statistics"
        print(f"\n{title} Jobs:")
        print("-" * 60)
        
        for model in models:
            print(f"\n  Model: {model}")
            for exp in experiments:
                job = jobs[job_type][model][exp]
                print(f"    {exp}:")
                print(f"      Status:   {job.symbol} {job.status}")
                if job.job_id:
                    print(f"      Job ID:   {job.job_id}")
                if job.start_time:
                    print(f"      Started:  {job.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                if job.end_time:
                    print(f"      Ended:    {job.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                if job.duration:
                    print(f"      Duration: {job.duration_str}")
                if job.error_message:
                    print(f"      Error:    {job.error_message[:80]}...")


def get_experiments_and_models(slurm_dir: Path) -> Tuple[List[str], List[str]]:
    """Get list of experiments and models from manifest files."""
    experiments = set()
    models = set()
    
    manifests = list(slurm_dir.glob("manifest_*.yaml"))
    
    for manifest_path in manifests:
        exp_name = manifest_path.stem.replace('manifest_', '')
        experiments.add(exp_name)
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
            
            for script_info in manifest.get('scripts', []):
                if 'model' in script_info:
                    models.add(script_info['model'])
        except Exception:
            pass
    
    return sorted(list(experiments)), sorted(list(models))


def main():
    parser = argparse.ArgumentParser(
        description="Check progress of generative evaluation jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python check_progress.py                    # Show all experiments
    python check_progress.py -e exp_n10_c100    # Single experiment
    python check_progress.py --detailed         # Show timing details
    python check_progress.py --summary          # Summary statistics only

Status Symbols:
    ✓  Completed - output file exists
    ⟳  Running   - job is running in SLURM
    ⏳  Pending   - job is queued in SLURM
    ✗  Failed    - error log contains errors
    -  Not started
        """
    )
    
    parser.add_argument("--experiment", "-e", nargs='*', 
                       help="Experiment name(s) to check (default: all)")
    parser.add_argument("--model", "-m", nargs='*',
                       help="Model name(s) to check (default: all)")
    parser.add_argument("--detailed", "-d", action="store_true",
                       help="Show detailed timing information")
    parser.add_argument("--summary", "-s", action="store_true",
                       help="Show summary statistics only")
    parser.add_argument("--output-dir", 
                       default=DEFAULT_OUTPUT_DIR,
                       help=f"Output base directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--log-dir",
                       default=DEFAULT_LOG_DIR,
                       help=f"Log directory (default: {DEFAULT_LOG_DIR})")
    parser.add_argument("--slurm-dir",
                       help="SLURM scripts directory (default: auto-detect from script location)")
    parser.add_argument("--gen-only", action="store_true",
                       help="Show only generation jobs")
    parser.add_argument("--stats-only", action="store_true",
                       help="Show only statistics jobs")
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent.parent
    slurm_dir = Path(args.slurm_dir) if args.slurm_dir else script_dir / "slurm_scripts"
    output_base = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    
    # Get experiments and models
    all_experiments, all_models = get_experiments_and_models(slurm_dir)
    
    if not all_experiments:
        print(f"No experiments found in {slurm_dir}")
        print("Run generate_slurm.py first to create experiment manifests.")
        return
    
    # Filter by arguments
    if args.experiment:
        experiments = [e for e in args.experiment if e in all_experiments]
        if not experiments:
            print(f"No matching experiments found. Available: {all_experiments}")
            return
    else:
        experiments = all_experiments
    
    if args.model:
        models = [m for m in args.model if m in all_models]
        if not models:
            print(f"No matching models found. Available: {all_models}")
            return
    else:
        models = all_models
    
    print(f"Checking progress...")
    print(f"  Log directory:    {log_dir}")
    print(f"  Output directory: {output_base}")
    print(f"  Models:           {len(models)}")
    print(f"  Experiments:      {len(experiments)}")
    
    # Collect all job information
    jobs = collect_all_jobs(models, experiments, output_base, log_dir)
    
    # Display based on options
    if args.summary:
        print_summary(jobs, models, experiments)
    elif args.detailed:
        print_detailed_table(jobs, models, experiments)
    else:
        # Default: show tables
        if not args.stats_only:
            print_table(jobs, models, experiments, 'gen', show_details=True)
        if not args.gen_only:
            print_table(jobs, models, experiments, 'stats', show_details=True)
        print_summary(jobs, models, experiments)


if __name__ == "__main__":
    main()
