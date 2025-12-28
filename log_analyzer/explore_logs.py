#!/usr/bin/env python3
"""
Interactive Log Explorer
========================
Search and explore specific logs interactively.
"""

import os
import sys
import re
import argparse
from datetime import datetime
from collections import defaultdict


def search_logs(directories, pattern, file_type=None, limit=50):
    """Search for logs matching a pattern."""
    regex = re.compile(pattern, re.IGNORECASE)
    matches = []
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        for entry in os.scandir(directory):
            if entry.is_file():
                if file_type and not entry.name.endswith(f'.{file_type}'):
                    continue
                    
                if regex.search(entry.name):
                    matches.append({
                        'path': entry.path,
                        'name': entry.name,
                        'size': entry.stat().st_size,
                        'mtime': datetime.fromtimestamp(entry.stat().st_mtime),
                    })
                    
                    if len(matches) >= limit:
                        break
        
        if len(matches) >= limit:
            break
    
    return matches


def read_log_content(filepath, head=50, tail=50, grep=None):
    """Read log content with options."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if grep:
            pattern = re.compile(grep, re.IGNORECASE)
            lines = [l for l in lines if pattern.search(l)]
            return lines[:100]  # Limit grep results
        
        result = []
        if head and len(lines) > head + tail:
            result.extend(lines[:head])
            result.append(f"\n... [{len(lines) - head - tail} lines omitted] ...\n\n")
            result.extend(lines[-tail:])
        else:
            result = lines
            
        return result
        
    except Exception as e:
        return [f"Error reading file: {e}"]


def find_job_pairs(directories, job_id=None, job_name=None):
    """Find matching .out and .err files for a job."""
    matches = defaultdict(dict)
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        for entry in os.scandir(directory):
            if not entry.is_file():
                continue
                
            name = entry.name
            if job_id and job_id not in name:
                continue
            if job_name and job_name.lower() not in name.lower():
                continue
            
            # Extract base name without extension
            if name.endswith('.out'):
                base = name[:-4]
                matches[base]['out'] = entry.path
            elif name.endswith('.err'):
                base = name[:-4]
                matches[base]['err'] = entry.path
    
    return dict(matches)


def analyze_single_job(out_path=None, err_path=None):
    """Provide detailed analysis of a single job."""
    analysis = {
        'status': 'unknown',
        'start_time': None,
        'end_time': None,
        'duration': None,
        'errors': [],
        'warnings': [],
        'results': [],
    }
    
    # Parse .out file
    if out_path and os.path.exists(out_path):
        with open(out_path, 'r', errors='ignore') as f:
            content = f.read()
            
        # Look for dates
        date_pattern = re.compile(
            r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+'
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+'
            r'\d{1,2}\s+\d{2}:\d{2}:\d{2}'
        )
        dates = date_pattern.findall(content)
        if dates:
            analysis['start_time'] = 'Found'
            if len(dates) > 1:
                analysis['end_time'] = 'Found'
                
        if 'ended successfully' in content.lower() or 'completed' in content.lower():
            analysis['status'] = 'completed'
    
    # Parse .err file
    if err_path and os.path.exists(err_path):
        with open(err_path, 'r', errors='ignore') as f:
            lines = f.readlines()
        
        for line in lines:
            if 'error' in line.lower():
                analysis['errors'].append(line.strip()[:100])
            if 'warning' in line.lower():
                analysis['warnings'].append(line.strip()[:100])
            if 'RESULT_ROW' in line:
                analysis['results'].append(line.strip()[:200])
        
        # Limit
        analysis['errors'] = analysis['errors'][:10]
        analysis['warnings'] = analysis['warnings'][:10]
    
    return analysis


def interactive_mode(directories):
    """Run interactive exploration mode."""
    print("\n=== SLURM Log Explorer ===")
    print("Commands:")
    print("  search <pattern>  - Search for logs matching pattern")
    print("  job <id>          - Find all logs for a job ID")
    print("  read <path>       - Read a log file")
    print("  grep <pattern> <path> - Search within a file")
    print("  analyze <path>    - Analyze a specific job")
    print("  quit              - Exit")
    print()
    
    while True:
        try:
            cmd = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if not cmd:
            continue
            
        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if action == 'quit' or action == 'exit':
            break
            
        elif action == 'search':
            matches = search_logs(directories, args)
            print(f"\nFound {len(matches)} matches:")
            for m in matches[:20]:
                print(f"  {m['name']} ({m['size']/1024:.1f} KB)")
                
        elif action == 'job':
            pairs = find_job_pairs(directories, job_id=args)
            print(f"\nFound {len(pairs)} job file sets:")
            for base, files in list(pairs.items())[:20]:
                print(f"  {base}:")
                for ext, path in files.items():
                    print(f"    .{ext}: {path}")
                    
        elif action == 'read':
            lines = read_log_content(args)
            print("".join(lines))
            
        elif action == 'grep':
            grep_parts = args.split(maxsplit=1)
            if len(grep_parts) == 2:
                lines = read_log_content(grep_parts[1], grep=grep_parts[0])
                print("".join(lines))
            else:
                print("Usage: grep <pattern> <filepath>")
                
        elif action == 'analyze':
            # Try to find both .out and .err
            if args.endswith('.out'):
                out_path = args
                err_path = args[:-4] + '.err'
            elif args.endswith('.err'):
                err_path = args
                out_path = args[:-4] + '.out'
            else:
                out_path = args + '.out'
                err_path = args + '.err'
                
            analysis = analyze_single_job(
                out_path if os.path.exists(out_path) else None,
                err_path if os.path.exists(err_path) else None,
            )
            
            print(f"\nJob Analysis:")
            print(f"  Status: {analysis['status']}")
            print(f"  Start time: {analysis['start_time']}")
            print(f"  End time: {analysis['end_time']}")
            print(f"  Errors: {len(analysis['errors'])}")
            print(f"  Warnings: {len(analysis['warnings'])}")
            print(f"  Results: {len(analysis['results'])}")
            
            if analysis['errors']:
                print("\n  Sample errors:")
                for e in analysis['errors'][:3]:
                    print(f"    {e}")
            
            if analysis['results']:
                print("\n  Results:")
                for r in analysis['results']:
                    print(f"    {r}")
        else:
            print(f"Unknown command: {action}")


def main():
    parser = argparse.ArgumentParser(description='Explore SLURM logs')
    parser.add_argument('--dirs', nargs='+',
                       default=['/gpfs/ostor/ossc9424/logs3',
                               '/gpfs/ostor/ossc9424/logs2',
                               '/gpfs/ostor/ossc9424/logs'],
                       help='Log directories')
    parser.add_argument('--search', '-s', help='Search pattern')
    parser.add_argument('--job', '-j', help='Find job by ID')
    parser.add_argument('--read', '-r', help='Read a file')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.search:
        matches = search_logs(args.dirs, args.search)
        for m in matches:
            print(f"{m['name']}\t{m['size']}\t{m['path']}")
            
    elif args.job:
        pairs = find_job_pairs(args.dirs, job_id=args.job)
        for base, files in pairs.items():
            print(f"{base}:")
            for ext, path in files.items():
                print(f"  {path}")
                
    elif args.read:
        lines = read_log_content(args.read)
        print("".join(lines))
        
    elif args.interactive or len(sys.argv) == 1:
        interactive_mode(args.dirs)
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
