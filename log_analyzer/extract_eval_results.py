#!/usr/bin/env python3
"""
Evaluation Results Extractor
=============================
Specifically extracts and summarizes evaluation results from logs.
"""

import os
import re
import json
import csv
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def parse_eval_log(filepath):
    """Parse an evaluation log file."""
    result = {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'task_name': None,
        'task_type': None,  # 'static' or 'finetune'
        'data_type': None,
        'model': None,
        'mode': None,  # 'train' or 'test'
        'has_result': False,
        'result_row': None,
        'metrics': {},
        'start_time': None,
        'end_time': None,
        'error': None,
    }
    
    filename = result['filename']
    
    # Detect task type from filename
    if '-ft-' in filename or 'finetune' in filename.lower():
        result['task_type'] = 'finetune'
    else:
        result['task_type'] = 'static'
    
    # Extract model info
    if 'BASE' in filename or '80M' in filename:
        result['model'] = 'BASE'
    elif 'small' in filename.lower():
        result['model'] = 'small'
    elif 'medium2x' in filename.lower():
        result['model'] = 'medium2x'
    elif 'medium' in filename.lower():
        result['model'] = 'medium'
    elif 'Gen-' in filename:
        match = re.search(r'Gen-(\w+)', filename)
        if match:
            result['model'] = f'Gen-{match.group(1)}'
    
    # Extract mode
    if 'test' in filename.lower():
        result['mode'] = 'test'
    else:
        result['mode'] = 'train'
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Parse first line for task info: 'task_name' [data_type]
        if lines:
            first_line = lines[0]
            task_match = re.search(r"'([^']+)'(?:\s*\[([^\]]+)\])?", first_line)
            if task_match:
                result['task_name'] = task_match.group(1)
                result['data_type'] = task_match.group(2)
        
        # Look for RESULT_ROW in last lines
        for line in reversed(lines[-30:]):
            if 'RESULT_ROW' in line:
                result['result_row'] = line.strip()
                
                # Parse the result row
                parts = line.split('RESULT_ROW')[-1].strip().split(',')
                if len(parts) > 2:
                    result['has_result'] = True
                    result['mode'] = parts[0].strip() if parts else result['mode']
                    
                    # Try to extract metrics (usually at the end)
                    for part in parts:
                        part = part.strip()
                        # Look for numeric values that could be metrics
                        if re.match(r'^[\d.]+$', part):
                            if 'metric1' not in result['metrics']:
                                result['metrics']['metric1'] = float(part)
                            elif 'metric2' not in result['metrics']:
                                result['metrics']['metric2'] = float(part)
                break
        
        # Extract timestamps
        date_pattern = re.compile(r'^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})')
        for line in lines[:10]:
            match = date_pattern.match(line)
            if match:
                result['start_time'] = f"{match.group(1)}-{match.group(2)}-{match.group(3)} {match.group(4)}:{match.group(5)}:{match.group(6)}"
                break
        
        for line in reversed(lines[-10:]):
            match = date_pattern.match(line)
            if match:
                result['end_time'] = f"{match.group(1)}-{match.group(2)}-{match.group(3)} {match.group(4)}:{match.group(5)}:{match.group(6)}"
                break
                
    except Exception as e:
        result['error'] = str(e)
    
    return result


def find_eval_logs(directories):
    """Find all evaluation log files."""
    files = []
    eval_patterns = [
        re.compile(r'eval', re.IGNORECASE),
        re.compile(r'static', re.IGNORECASE),
        re.compile(r'-ft-', re.IGNORECASE),
        re.compile(r'finetune', re.IGNORECASE),
        re.compile(r'test', re.IGNORECASE),
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.endswith('.err'):
                for pattern in eval_patterns:
                    if pattern.search(entry.name):
                        files.append(entry.path)
                        break
    
    return files


def summarize_results(logs):
    """Generate summary of evaluation results."""
    summary = {
        'total_eval_logs': len(logs),
        'with_results': sum(1 for l in logs if l['has_result']),
        'without_results': sum(1 for l in logs if not l['has_result']),
        'by_task': defaultdict(lambda: {'total': 0, 'with_results': 0, 'results': []}),
        'by_model': defaultdict(lambda: {'total': 0, 'with_results': 0}),
        'by_task_type': defaultdict(lambda: {'total': 0, 'with_results': 0}),
        'successful_results': [],
    }
    
    for log in logs:
        task = log['task_name'] or 'unknown'
        model = log['model'] or 'unknown'
        task_type = log['task_type'] or 'unknown'
        
        summary['by_task'][task]['total'] += 1
        summary['by_model'][model]['total'] += 1
        summary['by_task_type'][task_type]['total'] += 1
        
        if log['has_result']:
            summary['by_task'][task]['with_results'] += 1
            summary['by_model'][model]['with_results'] += 1
            summary['by_task_type'][task_type]['with_results'] += 1
            summary['by_task'][task]['results'].append({
                'model': model,
                'mode': log['mode'],
                'result_row': log['result_row'],
            })
            summary['successful_results'].append({
                'task': task,
                'model': model,
                'mode': log['mode'],
                'result_row': log['result_row'],
            })
    
    return summary


def export_results_csv(logs, output_file):
    """Export results to CSV for further analysis."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'filename', 'task_name', 'task_type', 'data_type', 
            'model', 'mode', 'has_result', 'result_row'
        ])
        
        for log in logs:
            writer.writerow([
                log['filename'],
                log['task_name'],
                log['task_type'],
                log['data_type'],
                log['model'],
                log['mode'],
                log['has_result'],
                log['result_row'],
            ])


def main():
    parser = argparse.ArgumentParser(description='Extract evaluation results')
    parser.add_argument('--dirs', nargs='+',
                       default=['/gpfs/ostor/ossc9424/logs3',
                               '/gpfs/ostor/ossc9424/logs2',
                               '/gpfs/ostor/ossc9424/logs'],
                       help='Log directories')
    parser.add_argument('--output', '-o', default='eval_results',
                       help='Output prefix')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    print("Finding evaluation log files...")
    files = find_eval_logs(args.dirs)
    print(f"Found {len(files)} potential evaluation log files.")
    
    print("Parsing logs...")
    workers = args.workers or min(mp.cpu_count(), 16)
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        logs = list(executor.map(parse_eval_log, files))
    
    print("Generating summary...")
    summary = summarize_results(logs)
    
    # Save JSON
    output_json = f"{args.output}.json"
    with open(output_json, 'w') as f:
        # Convert defaultdicts to regular dicts
        output = {
            'total_eval_logs': summary['total_eval_logs'],
            'with_results': summary['with_results'],
            'without_results': summary['without_results'],
            'by_task': dict(summary['by_task']),
            'by_model': dict(summary['by_model']),
            'by_task_type': dict(summary['by_task_type']),
            'successful_results': summary['successful_results'][:100],  # First 100
        }
        json.dump(output, f, indent=2, default=str)
    
    # Save CSV
    output_csv = f"{args.output}.csv"
    export_results_csv(logs, output_csv)
    
    print(f"\nResults saved to {output_json} and {output_csv}")
    print("\nSummary:")
    print(f"  Total evaluation logs: {summary['total_eval_logs']}")
    print(f"  With results: {summary['with_results']}")
    print(f"  Without results: {summary['without_results']}")
    print(f"\nBy task type:")
    for task_type, stats in summary['by_task_type'].items():
        print(f"  {task_type}: {stats['total']} total, {stats['with_results']} with results")
    print(f"\nBy model:")
    for model, stats in sorted(summary['by_model'].items()):
        print(f"  {model}: {stats['total']} total, {stats['with_results']} with results")
    print(f"\nBy task (top 10):")
    sorted_tasks = sorted(summary['by_task'].items(), key=lambda x: -x[1]['total'])
    for task, stats in sorted_tasks[:10]:
        print(f"  {task}: {stats['total']} total, {stats['with_results']} with results")


if __name__ == '__main__':
    main()
