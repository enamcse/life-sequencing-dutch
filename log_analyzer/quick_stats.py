#!/usr/bin/env python3
"""
Quick Statistics Script
=======================
Quickly scan log directories and provide basic statistics without full parsing.
Useful for getting an overview before running the full analysis.
"""

import os
import sys
from collections import defaultdict
from datetime import datetime
import re


def quick_scan(directories):
    """Quickly scan directories and gather statistics."""
    stats = {
        'total_files': 0,
        'total_size_bytes': 0,
        'out_files': 0,
        'err_files': 0,
        'other_files': 0,
        'by_directory': {},
        'file_sizes': [],
        'job_name_samples': [],
    }
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
            
        dir_stats = {
            'files': 0,
            'size': 0,
            'out': 0,
            'err': 0,
        }
        
        print(f"Scanning {directory}...")
        try:
            for entry in os.scandir(directory):
                if entry.is_file():
                    stats['total_files'] += 1
                    dir_stats['files'] += 1
                    
                    size = entry.stat().st_size
                    stats['total_size_bytes'] += size
                    dir_stats['size'] += size
                    stats['file_sizes'].append(size)
                    
                    if entry.name.endswith('.out'):
                        stats['out_files'] += 1
                        dir_stats['out'] += 1
                    elif entry.name.endswith('.err'):
                        stats['err_files'] += 1
                        dir_stats['err'] += 1
                    else:
                        stats['other_files'] += 1
                    
                    if len(stats['job_name_samples']) < 20:
                        stats['job_name_samples'].append(entry.name)
        except PermissionError:
            print(f"  Permission denied: {directory}")
            
        stats['by_directory'][directory] = dir_stats
    
    return stats


def format_size(size_bytes):
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def print_stats(stats):
    """Print statistics in a nice format."""
    print("\n" + "=" * 60)
    print("QUICK LOG DIRECTORY STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal files: {stats['total_files']:,}")
    print(f"Total size: {format_size(stats['total_size_bytes'])}")
    print(f"  .out files: {stats['out_files']:,}")
    print(f"  .err files: {stats['err_files']:,}")
    print(f"  other files: {stats['other_files']:,}")
    
    print("\nBy Directory:")
    for directory, dir_stats in stats['by_directory'].items():
        print(f"  {directory}:")
        print(f"    Files: {dir_stats['files']:,}")
        print(f"    Size: {format_size(dir_stats['size'])}")
        print(f"    .out: {dir_stats['out']:,}, .err: {dir_stats['err']:,}")
    
    if stats['file_sizes']:
        avg_size = sum(stats['file_sizes']) / len(stats['file_sizes'])
        min_size = min(stats['file_sizes'])
        max_size = max(stats['file_sizes'])
        print(f"\nFile Size Statistics:")
        print(f"  Average: {format_size(avg_size)}")
        print(f"  Min: {format_size(min_size)}")
        print(f"  Max: {format_size(max_size)}")
    
    if stats['job_name_samples']:
        print(f"\nSample filenames:")
        for name in stats['job_name_samples'][:10]:
            print(f"  {name}")
    
    print("\n" + "=" * 60)


def main():
    directories = [
        '/gpfs/ostor/ossc9424/logs3',
        '/gpfs/ostor/ossc9424/logs2',
        '/gpfs/ostor/ossc9424/logs',
    ]
    
    # Allow command line override
    if len(sys.argv) > 1:
        directories = sys.argv[1:]
    
    stats = quick_scan(directories)
    print_stats(stats)
    
    # Estimate processing time
    files_per_second = 500  # Approximate
    estimated_time = stats['total_files'] / files_per_second
    print(f"\nEstimated full analysis time: {estimated_time/60:.1f} minutes")
    print("(Run with --workers N to use multiple CPU cores)")


if __name__ == '__main__':
    main()
