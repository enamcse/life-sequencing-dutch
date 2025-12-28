#!/usr/bin/env python3
"""
SLURM Log Analyzer for Large-Scale Log Processing
==================================================
Efficiently analyzes ~8GB of SLURM logs across multiple directories.

Features:
- Streaming file processing (memory efficient)
- Parallel processing for speed
- Extracts job metadata, timings, and insights
- Generates summaries by job type, date, model, etc.
"""

import os
import re
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


# ============================================================================
# Date/Time Parsing Utilities
# ============================================================================

# Pattern for .out files: "Sat Dec 27 20:00:54 CET 2025"
OUT_DATE_PATTERN = re.compile(
    r'^(?:Started:|End:|Completed:)?\s*'
    r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+'
    r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+'
    r'(\d{1,2})\s+'
    r'(\d{2}):(\d{2}):(\d{2})\s+'
    r'(?:CET|CEST|UTC)?\s*'
    r'(\d{4})',
    re.IGNORECASE
)

# Pattern for .err files: "2025-12-27 20:01:00"
ERR_DATE_PATTERN = re.compile(
    r'^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})'
)

MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}


def parse_out_date(line: str) -> Optional[datetime]:
    """Parse date from .out file format."""
    match = OUT_DATE_PATTERN.search(line)
    if match:
        try:
            month = MONTH_MAP[match.group(2).lower()]
            return datetime(
                year=int(match.group(7)),
                month=month,
                day=int(match.group(3)),
                hour=int(match.group(4)),
                minute=int(match.group(5)),
                second=int(match.group(6))
            )
        except (ValueError, KeyError):
            return None
    return None


def parse_err_date(line: str) -> Optional[datetime]:
    """Parse date from .err file format."""
    match = ERR_DATE_PATTERN.match(line)
    if match:
        try:
            return datetime(
                year=int(match.group(1)),
                month=int(match.group(2)),
                day=int(match.group(3)),
                hour=int(match.group(4)),
                minute=int(match.group(5)),
                second=int(match.group(6))
            )
        except ValueError:
            return None
    return None


# ============================================================================
# Job Classification
# ============================================================================

# Model size patterns
MODEL_PATTERNS = {
    'small': re.compile(r'small|2M|3M', re.I),
    'medium': re.compile(r'medium(?!2x)|8M', re.I),
    'medium2x': re.compile(r'medium2x|15M', re.I),
    'BASE': re.compile(r'BASE|large|80M', re.I),
    'cceff': re.compile(r'cceff|160M', re.I),
    'ccall': re.compile(r'ccall|540M', re.I),
}

# Job type patterns
JOB_TYPE_PATTERNS = {
    'pretrain': re.compile(r'pretrain', re.I),
    'finetune': re.compile(r'finetune|ft[-_]', re.I),
    'inference': re.compile(r'infer|embedding', re.I),
    'evaluation': re.compile(r'eval|static|test', re.I),
    'generative': re.compile(r'generative|gen[-_]', re.I),
    'preprocess': re.compile(r'preprocess|export|trie|pipeline', re.I),
}

# Dataset patterns
DATASET_PATTERNS = {
    'D4': re.compile(r'D4', re.I),
    'D3': re.compile(r'D3', re.I),
}

# Masking patterns
MASKING_PATTERNS = {
    'event': re.compile(r'event', re.I),
    'random': re.compile(r'random', re.I),
}


def classify_job(filename: str, content_sample: str = "") -> Dict[str, str]:
    """Classify job based on filename and content."""
    result = {
        'model_size': 'unknown',
        'job_type': 'unknown',
        'dataset': 'D3',  # Default
        'masking': 'unknown',
        'is_test': 'test' in filename.lower(),
    }
    
    combined = filename + " " + content_sample
    
    # Model size
    for size, pattern in MODEL_PATTERNS.items():
        if pattern.search(combined):
            result['model_size'] = size
            break
    
    # Job type
    for jtype, pattern in JOB_TYPE_PATTERNS.items():
        if pattern.search(combined):
            result['job_type'] = jtype
            break
    
    # Dataset
    for ds, pattern in DATASET_PATTERNS.items():
        if pattern.search(combined):
            result['dataset'] = ds
            break
    
    # Masking
    for mask, pattern in MASKING_PATTERNS.items():
        if pattern.search(combined):
            result['masking'] = mask
            break
    
    return result


# ============================================================================
# Log Entry Data Structure
# ============================================================================

@dataclass
class JobLog:
    """Represents a parsed SLURM job log."""
    job_id: str
    job_name: str
    file_path: str
    file_type: str  # 'out' or 'err'
    file_size: int
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Classification
    model_size: str = 'unknown'
    job_type: str = 'unknown'
    dataset: str = 'D3'
    masking: str = 'unknown'
    is_test: bool = False
    
    # Status
    has_result: bool = False
    result_mode: Optional[str] = None  # 'train' or 'test'
    completed_normally: bool = False
    
    # Insights
    task_name: Optional[str] = None
    num_ranks: Optional[int] = None
    model_params: Optional[str] = None
    error_summary: Optional[str] = None
    
    # Raw data for debugging
    first_lines: List[str] = field(default_factory=list)
    last_lines: List[str] = field(default_factory=list)


# ============================================================================
# File Parsing
# ============================================================================

def parse_filename(filename: str) -> Tuple[str, str]:
    """
    Extract job_id and job_name from filename.
    Handles both conventions:
    - job_id.job_name.ext
    - job_name-job_id.ext
    """
    base = filename.rsplit('.', 1)[0]  # Remove extension
    
    # Try job_id.job_name pattern first (job_id is pure number < 10000)
    parts = base.split('.', 1)
    if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) <= 10000:
        return parts[0], parts[1]
    
    # Try job_name-job_id pattern
    parts = base.rsplit('-', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1], parts[0]
    
    # Try job_name.job_id pattern
    parts = base.rsplit('.', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1], parts[0]
    
    # Fallback: use filename as job_name
    return "unknown", base


def read_file_ends(filepath: str, head_lines: int = 50, tail_lines: int = 50) -> Tuple[List[str], List[str]]:
    """Efficiently read first and last N lines of a file."""
    head = []
    tail = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            # Read head
            for i, line in enumerate(f):
                if i < head_lines:
                    head.append(line.rstrip())
                else:
                    break
            
            # For tail, we need to seek from end or use deque
            f.seek(0)
            from collections import deque
            tail = list(deque(f, maxlen=tail_lines))
            tail = [l.rstrip() for l in tail]
    except Exception as e:
        pass
    
    return head, tail


def extract_task_info(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Extract task name and type from error file first lines."""
    task_pattern = re.compile(r"'([^']+)'.*\[([^\]]+)\]|'([^']+)'")
    
    for line in lines[:10]:
        match = task_pattern.search(line)
        if match:
            if match.group(1):
                return match.group(1), match.group(2)
            elif match.group(3):
                return match.group(3), None
    return None, None


def extract_result_info(lines: List[str]) -> Tuple[bool, Optional[str]]:
    """Check if job produced results and extract mode."""
    result_pattern = re.compile(r'RESULT_ROW\s+(\w+)')
    
    for line in reversed(lines[-20:]):
        if 'RESULT_ROW' in line:
            match = result_pattern.search(line)
            if match:
                mode = match.group(1)
                # Check if it's just "RESULT_ROW" without actual data
                if len(line.split(',')) > 2:
                    return True, mode
            return False, None
    return False, None


def extract_num_ranks(lines: List[str]) -> Optional[int]:
    """Extract number of ranks/GPUs used."""
    rank_pattern = re.compile(r'(?:RANK|ranks?|gpus?|nodes?)[\s:=]+(\d+)', re.I)
    
    for line in lines:
        match = rank_pattern.search(line)
        if match:
            return int(match.group(1))
    return None


def extract_model_params(lines: List[str]) -> Optional[str]:
    """Extract model parameters info."""
    param_pattern = re.compile(r'(?:param(?:eter)?s?|total)[\s:=]*(\d+(?:\.\d+)?[MKB]?)', re.I)
    
    for line in lines:
        if 'param' in line.lower():
            match = param_pattern.search(line)
            if match:
                return match.group(1)
    return None


def parse_log_file(filepath: str) -> Optional[JobLog]:
    """Parse a single log file and extract all information."""
    try:
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath)
        
        # Determine file type
        if filename.endswith('.out'):
            file_type = 'out'
        elif filename.endswith('.err'):
            file_type = 'err'
        else:
            return None
        
        # Parse filename
        job_id, job_name = parse_filename(filename)
        
        # Read file ends
        head_lines, tail_lines = read_file_ends(filepath)
        
        # Initialize log entry
        log = JobLog(
            job_id=job_id,
            job_name=job_name,
            file_path=filepath,
            file_type=file_type,
            file_size=file_size,
            first_lines=head_lines[:5],
            last_lines=tail_lines[-5:],
        )
        
        # Extract timestamps based on file type
        if file_type == 'out':
            # Find first date in head
            for line in head_lines:
                dt = parse_out_date(line)
                if dt:
                    log.start_time = dt
                    break
            
            # Find last date in tail
            for line in reversed(tail_lines):
                dt = parse_out_date(line)
                if dt:
                    log.end_time = dt
                    break
                    
        else:  # err file
            # Find first date in head
            for line in head_lines:
                dt = parse_err_date(line)
                if dt:
                    log.start_time = dt
                    break
            
            # Find last date in tail
            for line in reversed(tail_lines):
                dt = parse_err_date(line)
                if dt:
                    log.end_time = dt
                    break
        
        # Calculate duration
        if log.start_time and log.end_time:
            delta = log.end_time - log.start_time
            log.duration_seconds = delta.total_seconds()
        
        # Classify job
        content_sample = " ".join(head_lines[:20])
        classification = classify_job(filename, content_sample)
        log.model_size = classification['model_size']
        log.job_type = classification['job_type']
        log.dataset = classification['dataset']
        log.masking = classification['masking']
        log.is_test = classification['is_test']
        
        # Extract additional info for err files
        if file_type == 'err':
            log.task_name, _ = extract_task_info(head_lines)
            log.has_result, log.result_mode = extract_result_info(tail_lines)
            log.model_params = extract_model_params(head_lines)
        
        # Extract common info
        log.num_ranks = extract_num_ranks(head_lines)
        
        # Check completion
        for line in tail_lines[-5:]:
            if any(x in line.lower() for x in ['completed', 'finished', 'ended successfully', 'done']):
                log.completed_normally = True
                break
        
        return log
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


# ============================================================================
# Parallel Processing
# ============================================================================

def get_all_log_files(directories: List[str]) -> List[str]:
    """Get all log files from directories."""
    files = []
    for directory in directories:
        if os.path.exists(directory):
            for entry in os.scandir(directory):
                if entry.is_file() and (entry.name.endswith('.out') or entry.name.endswith('.err')):
                    files.append(entry.path)
    return files


def process_files_parallel(files: List[str], num_workers: int = None) -> List[JobLog]:
    """Process files in parallel."""
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)
    
    logs = []
    total = len(files)
    
    print(f"Processing {total} files with {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(parse_log_file, f): f for f in files}
        
        for i, future in enumerate(as_completed(futures)):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{total} files...")
            
            result = future.result()
            if result:
                logs.append(result)
    
    print(f"Successfully parsed {len(logs)} log files.")
    return logs


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_by_month(logs: List[JobLog]) -> Dict[str, Dict]:
    """Analyze jobs by month."""
    monthly = defaultdict(lambda: {
        'count': 0,
        'job_types': defaultdict(int),
        'model_sizes': defaultdict(int),
        'total_duration_hours': 0,
        'completed': 0,
        'with_results': 0,
    })
    
    for log in logs:
        if log.start_time:
            key = log.start_time.strftime('%Y-%m')
            monthly[key]['count'] += 1
            monthly[key]['job_types'][log.job_type] += 1
            monthly[key]['model_sizes'][log.model_size] += 1
            if log.duration_seconds:
                monthly[key]['total_duration_hours'] += log.duration_seconds / 3600
            if log.completed_normally:
                monthly[key]['completed'] += 1
            if log.has_result:
                monthly[key]['with_results'] += 1
    
    # Convert defaultdicts to regular dicts for JSON serialization
    result = {}
    for month, data in sorted(monthly.items()):
        result[month] = {
            'count': data['count'],
            'job_types': dict(data['job_types']),
            'model_sizes': dict(data['model_sizes']),
            'total_duration_hours': round(data['total_duration_hours'], 2),
            'completed': data['completed'],
            'with_results': data['with_results'],
        }
    
    return result


def analyze_by_job_type(logs: List[JobLog]) -> Dict[str, Dict]:
    """Analyze jobs by type."""
    by_type = defaultdict(lambda: {
        'count': 0,
        'total_duration_hours': 0,
        'avg_duration_minutes': 0,
        'durations': [],
        'model_sizes': defaultdict(int),
        'completed': 0,
        'with_results': 0,
    })
    
    for log in logs:
        key = log.job_type
        by_type[key]['count'] += 1
        by_type[key]['model_sizes'][log.model_size] += 1
        if log.duration_seconds:
            by_type[key]['durations'].append(log.duration_seconds)
            by_type[key]['total_duration_hours'] += log.duration_seconds / 3600
        if log.completed_normally:
            by_type[key]['completed'] += 1
        if log.has_result:
            by_type[key]['with_results'] += 1
    
    # Calculate averages
    result = {}
    for jtype, data in by_type.items():
        durations = data['durations']
        result[jtype] = {
            'count': data['count'],
            'total_duration_hours': round(data['total_duration_hours'], 2),
            'avg_duration_minutes': round(sum(durations) / len(durations) / 60, 2) if durations else 0,
            'min_duration_minutes': round(min(durations) / 60, 2) if durations else 0,
            'max_duration_minutes': round(max(durations) / 60, 2) if durations else 0,
            'model_sizes': dict(data['model_sizes']),
            'completed': data['completed'],
            'completion_rate': round(data['completed'] / data['count'] * 100, 1) if data['count'] else 0,
            'with_results': data['with_results'],
        }
    
    return result


def analyze_pretrain_timing(logs: List[JobLog]) -> Dict[str, Dict]:
    """Analyze pretraining jobs by model size."""
    pretrain_logs = [l for l in logs if l.job_type == 'pretrain']
    
    by_model = defaultdict(lambda: {
        'count': 0,
        'durations': [],
        'datasets': defaultdict(int),
        'masking': defaultdict(int),
    })
    
    for log in pretrain_logs:
        key = log.model_size
        by_model[key]['count'] += 1
        by_model[key]['datasets'][log.dataset] += 1
        by_model[key]['masking'][log.masking] += 1
        if log.duration_seconds:
            by_model[key]['durations'].append(log.duration_seconds)
    
    result = {}
    for model, data in by_model.items():
        durations = data['durations']
        result[model] = {
            'count': data['count'],
            'avg_duration_hours': round(sum(durations) / len(durations) / 3600, 2) if durations else 0,
            'total_duration_hours': round(sum(durations) / 3600, 2) if durations else 0,
            'min_duration_hours': round(min(durations) / 3600, 2) if durations else 0,
            'max_duration_hours': round(max(durations) / 3600, 2) if durations else 0,
            'datasets': dict(data['datasets']),
            'masking': dict(data['masking']),
        }
    
    return result


def analyze_evaluation_results(logs: List[JobLog]) -> Dict[str, Any]:
    """Analyze evaluation jobs."""
    eval_logs = [l for l in logs if l.job_type in ('evaluation', 'finetune')]
    
    summary = {
        'total_eval_jobs': len(eval_logs),
        'with_results': sum(1 for l in eval_logs if l.has_result),
        'without_results': sum(1 for l in eval_logs if not l.has_result),
        'by_mode': defaultdict(int),
        'by_task': defaultdict(int),
        'by_model': defaultdict(int),
    }
    
    for log in eval_logs:
        if log.result_mode:
            summary['by_mode'][log.result_mode] += 1
        if log.task_name:
            summary['by_task'][log.task_name] += 1
        summary['by_model'][log.model_size] += 1
    
    summary['by_mode'] = dict(summary['by_mode'])
    summary['by_task'] = dict(summary['by_task'])
    summary['by_model'] = dict(summary['by_model'])
    
    return summary


def analyze_failures(logs: List[JobLog]) -> Dict[str, Any]:
    """Analyze failed or incomplete jobs."""
    incomplete = [l for l in logs if not l.completed_normally and l.duration_seconds]
    no_end_time = [l for l in logs if l.start_time and not l.end_time]
    
    return {
        'incomplete_count': len(incomplete),
        'no_end_time_count': len(no_end_time),
        'incomplete_by_type': dict(defaultdict(int, {l.job_type: 1 for l in incomplete})),
        'sample_incomplete': [
            {
                'job_name': l.job_name,
                'job_type': l.job_type,
                'model_size': l.model_size,
                'file': os.path.basename(l.file_path),
            }
            for l in incomplete[:10]
        ],
    }


def analyze_daily_activity(logs: List[JobLog]) -> Dict[str, int]:
    """Get job counts by date."""
    daily = defaultdict(int)
    
    for log in logs:
        if log.start_time:
            key = log.start_time.strftime('%Y-%m-%d')
            daily[key] += 1
    
    return dict(sorted(daily.items()))


def generate_summary_report(logs: List[JobLog]) -> str:
    """Generate a human-readable summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SLURM LOG ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Overall stats
    total_files = len(logs)
    out_files = sum(1 for l in logs if l.file_type == 'out')
    err_files = sum(1 for l in logs if l.file_type == 'err')
    total_size_gb = sum(l.file_size for l in logs) / (1024**3)
    
    lines.append("OVERALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total log files analyzed: {total_files}")
    lines.append(f"  - .out files: {out_files}")
    lines.append(f"  - .err files: {err_files}")
    lines.append(f"Total size: {total_size_gb:.2f} GB")
    lines.append("")
    
    # Jobs with timing info
    with_timing = [l for l in logs if l.duration_seconds]
    if with_timing:
        total_hours = sum(l.duration_seconds for l in with_timing) / 3600
        avg_hours = total_hours / len(with_timing)
        lines.append(f"Jobs with timing info: {len(with_timing)}")
        lines.append(f"Total compute time: {total_hours:.1f} hours ({total_hours/24:.1f} days)")
        lines.append(f"Average job duration: {avg_hours:.2f} hours")
        lines.append("")
    
    # By job type
    lines.append("JOBS BY TYPE")
    lines.append("-" * 40)
    by_type = analyze_by_job_type(logs)
    for jtype, data in sorted(by_type.items(), key=lambda x: -x[1]['count']):
        lines.append(f"  {jtype}: {data['count']} jobs, "
                    f"avg {data['avg_duration_minutes']:.1f} min, "
                    f"{data['completion_rate']:.0f}% completed")
    lines.append("")
    
    # By model size
    lines.append("JOBS BY MODEL SIZE")
    lines.append("-" * 40)
    model_counts = defaultdict(int)
    for log in logs:
        model_counts[log.model_size] += 1
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {model}: {count} jobs")
    lines.append("")
    
    # Pretraining analysis
    lines.append("PRETRAINING ANALYSIS")
    lines.append("-" * 40)
    pretrain = analyze_pretrain_timing(logs)
    for model, data in pretrain.items():
        lines.append(f"  {model}: {data['count']} runs, "
                    f"avg {data['avg_duration_hours']:.1f}h, "
                    f"total {data['total_duration_hours']:.1f}h")
    lines.append("")
    
    # Evaluation results
    lines.append("EVALUATION RESULTS")
    lines.append("-" * 40)
    eval_summary = analyze_evaluation_results(logs)
    lines.append(f"  Total evaluation jobs: {eval_summary['total_eval_jobs']}")
    lines.append(f"  Jobs with results: {eval_summary['with_results']}")
    lines.append(f"  Jobs without results: {eval_summary['without_results']}")
    lines.append("")
    
    # Monthly activity
    lines.append("MONTHLY ACTIVITY")
    lines.append("-" * 40)
    monthly = analyze_by_month(logs)
    for month, data in monthly.items():
        lines.append(f"  {month}: {data['count']} jobs, "
                    f"{data['total_duration_hours']:.1f}h compute time")
    lines.append("")
    
    # Failures
    lines.append("FAILURE ANALYSIS")
    lines.append("-" * 40)
    failures = analyze_failures(logs)
    lines.append(f"  Incomplete jobs: {failures['incomplete_count']}")
    lines.append(f"  Jobs without end time: {failures['no_end_time_count']}")
    lines.append("")
    
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze SLURM log files')
    parser.add_argument('--dirs', nargs='+', 
                       default=['/gpfs/ostor/ossc9424/logs3', 
                               '/gpfs/ostor/ossc9424/logs2',
                               '/gpfs/ostor/ossc9424/logs'],
                       help='Directories containing log files')
    parser.add_argument('--output', '-o', default='log_analysis_results',
                       help='Output directory for results')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--sample', type=int, default=None,
                       help='Only process this many files (for testing)')
    parser.add_argument('--cache', action='store_true',
                       help='Use cached parsed logs if available')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    cache_file = os.path.join(args.output, 'parsed_logs.pkl')
    
    # Check for cache
    if args.cache and os.path.exists(cache_file):
        print(f"Loading cached results from {cache_file}...")
        with open(cache_file, 'rb') as f:
            logs = pickle.load(f)
        print(f"Loaded {len(logs)} cached log entries.")
    else:
        # Get all files
        print("Scanning directories for log files...")
        files = get_all_log_files(args.dirs)
        print(f"Found {len(files)} log files.")
        
        if args.sample:
            import random
            files = random.sample(files, min(args.sample, len(files)))
            print(f"Sampling {len(files)} files for testing...")
        
        # Process files
        logs = process_files_parallel(files, args.workers)
        
        # Cache results
        print(f"Caching results to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(logs, f)
    
    # Generate analyses
    print("\nGenerating analysis...")
    
    # Monthly analysis
    monthly = analyze_by_month(logs)
    with open(os.path.join(args.output, 'monthly_analysis.json'), 'w') as f:
        json.dump(monthly, f, indent=2)
    
    # Job type analysis
    by_type = analyze_by_job_type(logs)
    with open(os.path.join(args.output, 'job_type_analysis.json'), 'w') as f:
        json.dump(by_type, f, indent=2)
    
    # Pretraining analysis
    pretrain = analyze_pretrain_timing(logs)
    with open(os.path.join(args.output, 'pretrain_analysis.json'), 'w') as f:
        json.dump(pretrain, f, indent=2)
    
    # Evaluation analysis
    eval_summary = analyze_evaluation_results(logs)
    with open(os.path.join(args.output, 'evaluation_analysis.json'), 'w') as f:
        json.dump(eval_summary, f, indent=2)
    
    # Daily activity
    daily = analyze_daily_activity(logs)
    with open(os.path.join(args.output, 'daily_activity.json'), 'w') as f:
        json.dump(daily, f, indent=2)
    
    # Failure analysis
    failures = analyze_failures(logs)
    with open(os.path.join(args.output, 'failure_analysis.json'), 'w') as f:
        json.dump(failures, f, indent=2)
    
    # Summary report
    report = generate_summary_report(logs)
    report_file = os.path.join(args.output, 'summary_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\nSummary report saved to {report_file}")
    print("\n" + report)
    
    # Export all parsed data for further analysis
    all_data = [asdict(log) for log in logs]
    # Convert datetime objects to strings
    for d in all_data:
        if d['start_time']:
            d['start_time'] = d['start_time'].isoformat()
        if d['end_time']:
            d['end_time'] = d['end_time'].isoformat()
    
    with open(os.path.join(args.output, 'all_logs_data.json'), 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\nAll results saved to {args.output}/")


if __name__ == '__main__':
    main()
