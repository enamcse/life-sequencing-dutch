# SLURM Log Analyzer

A comprehensive Python toolkit for efficiently analyzing large-scale SLURM job logs (~8GB across 31,000+ files) from your research computing cluster.

## 🎯 What This Tool Does

This toolkit was created to analyze your SLURM logs scattered across three directories:
- `/gpfs/ostor/ossc9424/logs3` (1.76 GB, 15,473 files)
- `/gpfs/ostor/ossc9424/logs2` (3.74 GB, 8,127 files)  
- `/gpfs/ostor/ossc9424/logs` (1.91 GB, 7,852 files)

**Total: ~8GB of logs, 31,452 files**

### Key Capabilities

✅ **Temporal Analysis**: Shows which months/dates you ran which jobs  
✅ **Duration Analysis**: How much time specific tasks took (pretrain, eval, etc.)  
✅ **Model Performance**: Pretraining duration by model size (small, medium, BASE, ccall, etc.)  
✅ **Result Extraction**: Extracts evaluation RESULT_ROW data to CSV  
✅ **Failure Detection**: Identifies incomplete jobs and errors  
✅ **Efficient Processing**: Handles 8GB in 10-20 minutes using parallel processing  
✅ **Multiple Outputs**: Text reports, JSON data, CSV exports, HTML visualizations

## 📁 Files in This Package

**Location**: `/home/ehassan/log_analyzer/`

| File | Size | Description |
|------|------|-------------|
| `README.md` | 14KB | **This documentation** - Complete tutorial and reference |
| `run_analysis.sh` | 3KB | **Main runner** - Execute all analyses locally |
| `run_analysis_slurm.sh` | 3KB | **SLURM job script** - Submit to cluster queue |
| `analyze_slurm_logs.py` | 28KB | **Core engine** - Main parallel processing and analysis |
| `quick_stats.py` | 4KB | **Quick overview** - Fast directory scan (seconds) |
| `explore_logs.py` | 9KB | **Interactive explorer** - Search and browse logs |
| `analyze_pretrain.py` | 9KB | **Pretrain analysis** - Extract epochs, loss, checkpoints |
| `extract_eval_results.py` | 10KB | **Eval extraction** - Parse RESULT_ROW lines to CSV |
| `generate_html_report.py` | 15KB | **HTML generator** - Create visual reports with charts |

**Total**: 9 files, ~93KB of code (Python + Bash)

## 📋 Table of Contents

1. [What This Tool Does](#what-this-tool-does)
2. [Files in This Package](#files-in-this-package)
3. [Overview](#overview)
4. [Quick Start](#quick-start)
5. [Step-by-Step Tutorial](#step-by-step-tutorial)
6. [Scripts Reference](#scripts-reference)
7. [File Naming Conventions](#file-naming-conventions)
8. [Job Classifications](#job-classifications)
9. [Output Files](#output-files)
10. [Performance Tips](#performance-tips)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This toolkit processes your SLURM job logs with these features:

- **No External Dependencies**: Uses only Python 3.7+ standard library
- **Memory Efficient**: Streaming file processing for large logs
- **Parallel Processing**: Multi-core support (8-16 workers recommended)
- **Smart Classification**: Automatically detects job type, model size, dataset (D3/D4), masking type
- **Multiple Date Formats**: Handles both `.out` (e.g., "Sat Dec 27 20:00:54 CET 2025") and `.err` (e.g., "2025-12-27 20:01:00") formats
- **Comprehensive Reports**: Generates text summaries, JSON data, CSV exports, and interactive HTML

### What You'll Get

After running the analysis, you'll have:
- **Monthly/daily job activity charts** - See when you ran different experiments
- **Duration statistics** - Average and total time per job type
- **Pretraining metrics** - Epochs completed, loss values, checkpoint info by model size
- **Evaluation results** - Extracted RESULT_ROW data in CSV format
- **Failure analysis** - Jobs that didn't complete or produce results
- **Interactive HTML dashboard** - Beautiful visualizations with charts

---

## Quick Start

### 🚀 Option 1: Quick Test (Recommended First)

```bash
cd /home/ehassan/log_analyzer

# Quick overview (takes ~30 seconds)
python quick_stats.py

# Test with 500 random files (takes ~2-3 minutes)
./run_analysis.sh --sample 500 --workers 4
```

### 🏃 Option 2: Full Analysis - Local Run

```bash
cd /home/ehassan/log_analyzer

# Process all ~31,000 files (takes ~15-20 minutes)
./run_analysis.sh --workers 16
```

### 🎯 Option 3: Full Analysis - SLURM Job (Recommended)

```bash
cd /home/ehassan/log_analyzer

# Submit to cluster queue
sbatch run_analysis_slurm.sh

# Monitor progress
squeue -u $USER
tail -f /projects/0/prjs1589/stonybrook/logs/log_analysis-*.out
```

### 📊 View Results

```bash
cd analysis_output

# Read summary
cat summary_report.txt

# Copy HTML report to view in browser
# (Use WinSCP or scp to download report.html to your local machine)
```

## Scripts Reference

### 1. `run_analysis.sh` - Main Runner Script
The main entry point that runs all analyses in sequence.

```bash
./run_analysis.sh [options]

Options:
  --workers, -w N    Number of parallel workers (default: 8)
  --sample, -s N     Only process N files (for testing)
  --output, -o DIR   Output directory (default: ./analysis_output)
  --help, -h         Show help
```

### 2. `run_analysis_slurm.sh` - SLURM Job Script
Submit to run the full analysis as a SLURM job:

```bash
sbatch run_analysis_slurm.sh
```

### 3. `analyze_slurm_logs.py` - Main Analyzer
Full analysis of all log files with parallel processing.

```bash
python analyze_slurm_logs.py --dirs /path/to/logs1 /path/to/logs2 --output results/
```

Options:
- `--dirs`: Log directories (default: /gpfs/ostor/ossc9424/logs{,2,3})
- `--output, -o`: Output directory
- `--workers, -w`: Number of parallel workers
- `--sample`: Process only N files (for testing)
- `--cache`: Use cached results if available

### 4. `quick_stats.py` - Quick Overview
Fast scan to get basic statistics without full parsing (runs in seconds).

```bash
python quick_stats.py /path/to/logs
```

### 5. `explore_logs.py` - Interactive Explorer
Search and explore logs interactively.

```bash
# Interactive mode
python explore_logs.py -i

# Search for files
python explore_logs.py --search "pretrain.*D4"

# Find job by ID
python explore_logs.py --job 1234
```

Interactive commands:
- `search <pattern>` - Search for logs matching pattern
- `job <id>` - Find all logs for a job ID
- `read <path>` - Read a log file
- `grep <pattern> <path>` - Search within a file
- `analyze <path>` - Analyze a specific job
- `quit` - Exit

### 6. `analyze_pretrain.py` - Pretraining Analysis
Detailed analysis of pretraining jobs.

```bash
python analyze_pretrain.py --output pretrain_results.json
```

Extracts:
- Duration by model size
- Epochs completed
- Final loss values
- Checkpoint info
- Forced stops vs. completions

### 7. `extract_eval_results.py` - Evaluation Results
Extracts evaluation results from logs.

```bash
python extract_eval_results.py --output eval_results
```

### 8. `generate_html_report.py` - Visual Report Generator
Generates an interactive HTML report with charts.

```bash
python generate_html_report.py --input analysis_output --output report.html
```

---

## Step-by-Step Tutorial

### Step 1: Initial Setup

```bash
# Navigate to the analyzer directory
cd /home/ehassan/log_analyzer

# Make all scripts executable
chmod +x *.sh *.py

# Verify the log directories exist
ls -la /gpfs/ostor/ossc9424/logs*
```

### Step 2: Quick Overview (Recommended First Step)

Get a quick overview without full parsing:

```bash
python quick_stats.py
```

This will show you:
- Total number of files in each directory
- Total size of logs
- File type breakdown (.out vs .err)
- Sample filenames
- Estimated processing time

### Step 3: Test with a Sample

Before processing all 31,000+ files, test with a small sample:

```bash
./run_analysis.sh --sample 500 --workers 4
```

This processes only 500 random files and validates everything works.

### Step 4: Run Full Analysis

**Option A: Run Locally (if you have an interactive session)**
```bash
./run_analysis.sh --workers 16
```

**Option B: Submit as SLURM Job (Recommended)**
```bash
sbatch run_analysis_slurm.sh
```

Monitor the job:
```bash
squeue -u $USER
tail -f /projects/0/prjs1589/stonybrook/logs/log_analysis-*.out
```

### Step 5: Review Results

After analysis completes, check the outputs:

```bash
cd analysis_output

# Read the summary report
cat summary_report.txt

# Open the HTML report (copy to local machine to view in browser)
# Or use a text browser
less report.html

# Check specific analysis files
cat monthly_analysis.json | python -m json.tool | less
cat job_type_analysis.json | python -m json.tool | less
```

### Step 6: Interactive Exploration

For detailed investigation of specific jobs:

```bash
python explore_logs.py -i
```

Example session:
```
>>> search pretrain.*D4.*BASE
Found 45 matches:
  1234.pretrain-D4-BASE-event.out (1.2 MB)
  ...

>>> job 1234
Found 1 job file sets:
  1234.pretrain-D4-BASE-event:
    .out: /gpfs/ostor/ossc9424/logs2/1234.pretrain-D4-BASE-event.out
    .err: /gpfs/ostor/ossc9424/logs2/1234.pretrain-D4-BASE-event.err

>>> read /gpfs/ostor/ossc9424/logs2/1234.pretrain-D4-BASE-event.err
```

### Step 7: Export for Further Analysis

The analysis creates several JSON and CSV files you can use in other tools:

```bash
# Load in Python
import json
with open('analysis_output/all_logs_data.json') as f:
    logs = json.load(f)

# Load in pandas
import pandas as pd
df = pd.read_csv('analysis_output/eval_results.csv')
```

---

## File Naming Conventions

The analyzer handles multiple naming conventions used in your logs:

| Convention | Example | Notes |
|------------|---------|-------|
| New: `job_id.job_name.ext` | `1234.pretrain-D4-medium.out` | Current convention |
| Old: `job_name-job_id.ext` | `pretrain-D4-medium-1234.out` | Legacy convention |
| FT: `job_id.test-ft-N.ext` | `4025.test-ft-6.err` | Fine-tuning tests |

**Note**: Job IDs can be duplicated across server resets (max observed: ~8608).

## Date Formats

The analyzer parses two date formats:

| File Type | Format | Example |
|-----------|--------|---------|
| `.out` files | Weekday Month Day Time TZ Year | `Sat Dec 27 20:00:54 CET 2025` |
| `.err` files | ISO-style | `2025-12-27 20:01:00 [INFO] ...` |

---

## Job Classifications

### Model Sizes
| Code | Parameters | Description |
|------|------------|-------------|
| `small` | 2-3M | Early experiments |
| `medium` | 8M | Standard medium |
| `medium2x` | 15M | Larger medium |
| `BASE/large` | 80M | Base model |
| `cceff` | 160M | Efficient large |
| `ccall` | 540M | Full large |

### Job Types
| Type | Description | File Pattern |
|------|-------------|--------------|
| `pretrain` | Model pretraining | `pretrain` in name |
| `finetune` | Fine-tuning tasks | `ft-`, `finetune` |
| `inference` | Embedding inference | `infer`, `embedding` |
| `evaluation` | Static evaluation | `eval`, `static` |
| `generative` | Generative model tasks | `generative`, `gen-` |
| `preprocess` | Data preprocessing | `preprocess`, `pipeline` |

### Datasets
| Code | Description |
|------|-------------|
| `D4` | Latest dataset |
| `D3` | Previous dataset (default if not specified) |

### Masking Types
| Type | Description |
|------|-------------|
| `event` | Event-based masking |
| `random` | Random masking |

---

## Output Files

After running the analysis, you'll find these files in `analysis_output/`:

| File | Description |
|------|-------------|
| `summary_report.txt` | Human-readable summary of all findings |
| `report.html` | Interactive HTML report with charts |
| `monthly_analysis.json` | Job counts and compute time by month |
| `daily_activity.json` | Job counts by date |
| `job_type_analysis.json` | Statistics by job type |
| `pretrain_analysis.json` | Pretraining job statistics |
| `pretrain_detailed.json` | Detailed pretraining metrics |
| `evaluation_analysis.json` | Evaluation job statistics |
| `eval_results.json` | Extracted evaluation results |
| `eval_results.csv` | Evaluation results (for Excel/pandas) |
| `failure_analysis.json` | Failed and incomplete jobs |
| `all_logs_data.json` | All parsed log data (large file) |
| `parsed_logs.pkl` | Cached parsed data (for --cache option) |
| `quick_stats.txt` | Quick statistics output |

---

## Performance Tips

1. **Test first**: Always run with `--sample 500` first to verify everything works
2. **Use caching**: After first run, use `--cache` to skip re-parsing when regenerating reports
3. **Adjust workers**: More workers = faster processing but more memory
   - 8 workers: ~15-20 min for 31k files
   - 16 workers: ~8-12 min for 31k files
4. **Submit as SLURM job**: For full analysis, use `sbatch run_analysis_slurm.sh`
5. **SSD helps**: If possible, run from local SSD rather than network storage

## Troubleshooting

### "Permission denied" errors
Ensure you have read access to all log directories:
```bash
ls -la /gpfs/ostor/ossc9424/logs*
```

### Memory issues
Reduce `--workers` or use `--sample` to process fewer files:
```bash
./run_analysis.sh --workers 4 --sample 5000
```

### Slow processing
- Use more workers if CPU allows
- Ensure log directories are on fast storage
- Use `--cache` after first run

### Missing Python
The scripts require Python 3.7+. On the cluster:
```bash
module load Python/3.9.6-GCCcore-11.2.0
# or
source /path/to/your/venv/bin/activate
```

### Jobs not classified correctly
Check if your job naming follows one of the expected patterns. You can extend the patterns in `analyze_slurm_logs.py`:
```python
JOB_TYPE_PATTERNS = {
    'pretrain': re.compile(r'pretrain', re.I),
    # Add your custom patterns here
}
```

---

## Example Output

### Summary Report (summary_report.txt)
```
======================================================================
SLURM LOG ANALYSIS REPORT
Generated: 2025-12-28 14:30:00
======================================================================

OVERALL STATISTICS
----------------------------------------
Total log files analyzed: 31,452
  - .out files: 15,726
  - .err files: 15,726
Total size: 7.41 GB

Jobs with timing info: 28,500
Total compute time: 15,234.5 hours (634.8 days)
Average job duration: 0.53 hours

JOBS BY TYPE
----------------------------------------
  pretrain: 1,234 jobs, avg 45.2 min, 78% completed
  evaluation: 8,567 jobs, avg 12.3 min, 92% completed
  finetune: 2,345 jobs, avg 23.1 min, 85% completed
  inference: 1,890 jobs, avg 8.5 min, 95% completed
  ...

MONTHLY ACTIVITY
----------------------------------------
  2025-01: 2,345 jobs, 567.89h compute time
  2025-02: 3,456 jobs, 789.12h compute time
  ...
```

### Monthly Analysis (JSON)
```json
{
  "2025-01": {
    "count": 2345,
    "job_types": {"pretrain": 123, "evaluation": 456},
    "total_duration_hours": 567.89,
    "completed": 2100
  }
}
```

---

## Requirements

- **Python 3.7+** (uses only standard library - no external dependencies)
- **Access to log directories** on the cluster
- **~2-4GB RAM** for full analysis with 8 workers

---

## Files in This Package

```
log_analyzer/
├── README.md                  # This documentation
├── run_analysis.sh            # Main runner script (local)
├── run_analysis_slurm.sh      # SLURM job script (cluster)
├── analyze_slurm_logs.py      # Main analysis engine
├── quick_stats.py             # Quick statistics (fast)
├── explore_logs.py            # Interactive explorer
├── analyze_pretrain.py        # Pretraining analysis
├── extract_eval_results.py    # Evaluation results extraction
├── generate_html_report.py    # HTML report generator
└── analysis_output/           # Output directory (created on run)
```

---

## License

Internal use only - Research Computing Team.

---

## Contact

For issues or feature requests, contact the development team.
