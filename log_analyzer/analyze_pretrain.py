#!/usr/bin/env python3
"""
Specific Analysis: Pretraining Jobs
====================================
Detailed analysis of pretraining jobs including:
- Duration by model size
- Epochs completed
- Training loss trends
- GPU utilization patterns
"""

import os
import re
import json
import argparse
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def parse_pretrain_log(filepath):
    """Parse a pretraining log file for detailed metrics."""
    result = {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'model_size': 'unknown',
        'dataset': 'unknown',
        'masking': 'unknown',
        'start_time': None,
        'end_time': None,
        'duration_hours': None,
        'epochs_completed': 0,
        'final_loss': None,
        'loss_history': [],
        'checkpoints_saved': [],
        'forced_stop': False,
        'completed': False,
        'error': None,
    }
    
    filename = result['filename']
    
    # Extract model info from filename
    if 'small' in filename.lower():
        result['model_size'] = 'small'
    elif 'medium2x' in filename.lower():
        result['model_size'] = 'medium2x'
    elif 'medium' in filename.lower():
        result['model_size'] = 'medium'
    elif 'BASE' in filename or 'large' in filename.lower() or '80M' in filename:
        result['model_size'] = 'BASE'
    elif 'cceff' in filename.lower() or '160M' in filename:
        result['model_size'] = 'cceff'
    elif 'ccall' in filename.lower() or '540M' in filename:
        result['model_size'] = 'ccall'
    
    if 'D4' in filename:
        result['dataset'] = 'D4'
    elif 'D3' in filename:
        result['dataset'] = 'D3'
    
    if 'event' in filename.lower():
        result['masking'] = 'event'
    elif 'random' in filename.lower():
        result['masking'] = 'random'
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Date patterns
        out_date = re.compile(r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})\s+(\d{2}):(\d{2}):(\d{2}).*?(\d{4})')
        err_date = re.compile(r'^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})')
        
        # Find dates
        for line in lines[:100]:
            if result['start_time'] is None:
                match = out_date.search(line) or err_date.search(line)
                if match:
                    result['start_time'] = line[:50]
                    break
        
        for line in reversed(lines[-100:]):
            match = out_date.search(line) or err_date.search(line)
            if match:
                result['end_time'] = line[:50]
                break
        
        # Look for loss values (common patterns)
        loss_pattern = re.compile(r'loss[:\s=]+(\d+\.?\d*)', re.IGNORECASE)
        for line in lines:
            match = loss_pattern.search(line)
            if match:
                try:
                    loss = float(match.group(1))
                    if loss < 100:  # Reasonable loss value
                        result['loss_history'].append(loss)
                except:
                    pass
        
        if result['loss_history']:
            result['final_loss'] = result['loss_history'][-1]
        
        # Look for epoch info
        epoch_pattern = re.compile(r'epoch[:\s=]+(\d+)', re.IGNORECASE)
        max_epoch = 0
        for line in lines:
            match = epoch_pattern.search(line)
            if match:
                epoch = int(match.group(1))
                max_epoch = max(max_epoch, epoch)
        result['epochs_completed'] = max_epoch
        
        # Look for checkpoints
        checkpoint_pattern = re.compile(r'sav(?:ed|ing).*checkpoint|checkpoint.*saved?', re.IGNORECASE)
        for line in lines:
            if checkpoint_pattern.search(line):
                result['checkpoints_saved'].append(line.strip()[:100])
        
        # Check completion status
        if 'ended successfully' in content.lower() or 'training completed' in content.lower():
            result['completed'] = True
        
        # Check for forced stop
        if 'sigterm' in content.lower() or 'cancelled' in content.lower() or 'timeout' in content.lower():
            result['forced_stop'] = True
            
    except Exception as e:
        result['error'] = str(e)
    
    # Limit list sizes
    result['loss_history'] = result['loss_history'][:100]  # Keep first 100
    result['checkpoints_saved'] = result['checkpoints_saved'][:20]
    
    return result


def find_pretrain_logs(directories):
    """Find all pretraining log files."""
    files = []
    pretrain_pattern = re.compile(r'pretrain', re.IGNORECASE)
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        for entry in os.scandir(directory):
            if entry.is_file() and pretrain_pattern.search(entry.name):
                files.append(entry.path)
    
    return files


def analyze_pretrain_logs(logs):
    """Generate summary statistics from pretrain logs."""
    summary = {
        'total_pretrain_jobs': len(logs),
        'by_model_size': defaultdict(list),
        'by_dataset': defaultdict(list),
        'by_masking': defaultdict(list),
        'completed': 0,
        'forced_stopped': 0,
        'with_loss_data': 0,
    }
    
    for log in logs:
        summary['by_model_size'][log['model_size']].append(log)
        summary['by_dataset'][log['dataset']].append(log)
        summary['by_masking'][log['masking']].append(log)
        
        if log['completed']:
            summary['completed'] += 1
        if log['forced_stop']:
            summary['forced_stopped'] += 1
        if log['loss_history']:
            summary['with_loss_data'] += 1
    
    # Generate model size statistics
    model_stats = {}
    for model, model_logs in summary['by_model_size'].items():
        losses = [l['final_loss'] for l in model_logs if l['final_loss']]
        epochs = [l['epochs_completed'] for l in model_logs if l['epochs_completed']]
        
        model_stats[model] = {
            'count': len(model_logs),
            'completed': sum(1 for l in model_logs if l['completed']),
            'forced_stopped': sum(1 for l in model_logs if l['forced_stop']),
            'avg_final_loss': sum(losses) / len(losses) if losses else None,
            'avg_epochs': sum(epochs) / len(epochs) if epochs else None,
            'max_epochs': max(epochs) if epochs else 0,
        }
    
    summary['model_statistics'] = model_stats
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze pretraining logs')
    parser.add_argument('--dirs', nargs='+',
                       default=['/gpfs/ostor/ossc9424/logs3',
                               '/gpfs/ostor/ossc9424/logs2',
                               '/gpfs/ostor/ossc9424/logs'],
                       help='Log directories')
    parser.add_argument('--output', '-o', default='pretrain_analysis.json',
                       help='Output file')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    print("Finding pretraining log files...")
    files = find_pretrain_logs(args.dirs)
    print(f"Found {len(files)} pretraining log files.")
    
    print("Parsing logs...")
    workers = args.workers or min(mp.cpu_count(), 16)
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        logs = list(executor.map(parse_pretrain_log, files))
    
    print("Generating summary...")
    summary = analyze_pretrain_logs(logs)
    
    # Prepare output
    output = {
        'summary': {
            'total_jobs': summary['total_pretrain_jobs'],
            'completed': summary['completed'],
            'forced_stopped': summary['forced_stopped'],
            'with_loss_data': summary['with_loss_data'],
        },
        'by_model_size': summary['model_statistics'],
        'detailed_logs': logs[:100],  # Keep first 100 for reference
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to {args.output}")
    print("\nSummary:")
    print(f"  Total pretraining jobs: {summary['total_pretrain_jobs']}")
    print(f"  Completed: {summary['completed']}")
    print(f"  Force stopped: {summary['forced_stopped']}")
    print(f"\nBy model size:")
    for model, stats in summary['model_statistics'].items():
        print(f"  {model}: {stats['count']} jobs, "
              f"{stats['completed']} completed, "
              f"avg epochs: {stats['avg_epochs']:.1f}" if stats['avg_epochs'] else f"  {model}: {stats['count']} jobs")


if __name__ == '__main__':
    main()
