#!/usr/bin/env python3
"""
Pipeline Supervisor v2 for Generative Evaluation

Enhanced supervisor with:
- Runtime GPU/node modification via config file re-reading
- Priority queue for experiments
- Improved completion detection with customizable markers
- Result aggregation to CSV
- Self-backup/versioning

Designed to run as a SLURM job on the work_env partition (login node).

Usage:
    # Submit as SLURM job (recommended)
    sbatch supervisor_v2.slurm
    
    # Or run directly
    python supervisor_v2.py --config supervisor_v2_config.yaml

Experiment Naming Convention:
    exp_n{n}_c{c}_h{h}_g{g}_k{k}_t{temp}_{model}_{dataset}
    
    Example: exp_n100_c100_h20_g100_k20_t0.8_GenBASE_GD0

Job Naming:
    {job_number:03d}.{experiment_name}
    
    Example: 001.exp_n100_c100_h20_g100_k20_t0.8_GenBASE_GD0

Features:
    1. Runtime config re-reading (GPUs, priorities)
    2. Priority-based job submission
    3. Customizable completion markers
    4. Result aggregation (3 CSVs)
    5. Self-backup on start
"""

import argparse
import copy
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_SLURM_DIR = Path(__file__).parent.parent / "slurm_scripts"
DEFAULT_OUTPUT_DIR = Path("/projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval")
DEFAULT_LOG_DIR = Path("/projects/0/prjs1589/stonybrook/logs")
DEFAULT_STATE_DIR = Path(__file__).parent.parent / "supervisor_state"
DEFAULT_EXPORT_DIR = Path("/projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/exports")
DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent / "config" / "registry.yaml"
DEFAULT_CONFIG_DIR = Path("/projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval/configs")


class JobStatus(Enum):
    """Status of a pipeline job."""
    NOT_STARTED = "not_started"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"  # For jobs below priority threshold


# ============================================================================
# Completion Markers Configuration
# ============================================================================

# Default completion markers - EDIT HERE to add new markers
COMPLETION_MARKERS = {
    'generation': [
        # Any of these patterns indicate successful generation
        ("Generation Complete!", "Completed:"),
        ("Generation Complete!",),
    ],
    'statistics': [
        # Any of these patterns indicate successful statistics
        ("Statistics Complete!", "Completed:"),
        ("Statistics Complete!",),
        ("All Steps Completed:",),  # Combined stats+plots job
    ],
    'plots': [
        ("Plots Complete!",),
        ("All Steps Completed:",),
    ],
}

# Error patterns to detect failures
ERROR_PATTERNS = [
    r'(CUDA out of memory)',
    r'(RuntimeError:.*)',
    r'(FileNotFoundError:.*)',
    r'(Exception:.*)',
    r'(Error:.*)',
    r'(Traceback \(most recent call last\))',
    r'(srun: error:.*)',
    r'(slurmstepd: error:.*)',
]


# ============================================================================
# Experiment Definition
# ============================================================================

@dataclass
class ExperimentParams:
    """Full experiment parameters."""
    n: int = 100           # num_people
    c: int = 100           # num_generations
    h: int = 20            # horizon
    g: int = 100           # gap (prefix_gap)
    k: int = 20            # top_k (use -1 for vocab size)
    t: float = 0.8         # temperature
    model: str = ""        # model name
    dataset:
    
    @property
    def experiment_name(self) -> str:
        """Generate full experiment name."""
        # Handle special k values
        k_str = "v" if self.k == -1 else str(self.k)
        # Replace dots in temperature
        t_str = str(self.t).replace(".", "")
        # Clean model name (remove special chars)
        model_clean = re.sub(r'[^a-zA-Z0-9]', '', self.model)
        
        return f"exp_n{self.n}_c{self.c}_h{self.h}_g{self.g}_k{k_str}_t{t_str}_{model_clean}_{self.dataset}"
    
    @classmethod
    def from_name(cls, name: str) -> 'ExperimentParams':
        """Parse experiment name back to params."""
        pattern = r'exp_n(\d+)_c(\d+)_h(\d+)_g(\d+)_k(\w+)_t(\d+)_([^_]+)_(\w+)'
        match = re.match(pattern, name)
        if not match:
            raise ValueError(f"Cannot parse experiment name: {name}")
        
        n, c, h, g, k_str, t_str, model, dataset = match.groups()
        k = -1 if k_str == 'v' else int(k_str)
        t = float(t_str[0] + '.' + t_str[1:]) if len(t_str) > 1 else float(t_str)
        
        return cls(
            n=int(n), c=int(c), h=int(h), g=int(g),
            k=k, t=t, model=model, dataset=dataset
        )
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class JobState:
    """State of a single job."""
    experiment: str
    job_type: str  # 'gen', 'stats'
    status: JobStatus = JobStatus.NOT_STARTED
    slurm_job_id: Optional[str] = None
    job_number: int = 0  # For job naming: 001, 002, etc.
    gpu_slot: Optional[str] = None
    submit_time: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    priority: int = 0  # Higher = more important
    
    @property
    def key(self) -> str:
        return f"{self.job_type}:{self.experiment}"
    
    @property
    def job_name(self) -> str:
        """SLURM job name: {number}.{experiment}"""
        return f"{self.job_number:03d}.{self.experiment}"
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['status'] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'JobState':
        d = d.copy()
        d['status'] = JobStatus(d['status'])
        return cls(**d)


@dataclass
class PipelineState:
    """Complete state of the pipeline."""
    experiments: List[str] = field(default_factory=list)
    jobs: Dict[str, JobState] = field(default_factory=dict)
    gpu_assignments: Dict[str, str] = field(default_factory=dict)
    job_counter: int = 0  # For unique job numbering
    last_update: Optional[str] = None
    start_time: Optional[str] = None
    config_version: int = 0  # Incremented when config changes
    
    def to_dict(self) -> dict:
        return {
            'experiments': self.experiments,
            'jobs': {k: v.to_dict() for k, v in self.jobs.items()},
            'gpu_assignments': self.gpu_assignments,
            'job_counter': self.job_counter,
            'last_update': self.last_update,
            'start_time': self.start_time,
            'config_version': self.config_version,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PipelineState':
        state = cls(
            experiments=d.get('experiments', []),
            gpu_assignments=d.get('gpu_assignments', {}),
            job_counter=d.get('job_counter', 0),
            last_update=d.get('last_update'),
            start_time=d.get('start_time'),
            config_version=d.get('config_version', 0),
        )
        for k, v in d.get('jobs', {}).items():
            state.jobs[k] = JobState.from_dict(v)
        return state


# ============================================================================
# Utility Functions
# ============================================================================

def run_command(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    """Run a shell command."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def get_slurm_jobs(user: str = None) -> Dict[str, Dict]:
    """Get all SLURM jobs for the user."""
    if user is None:
        user = os.environ.get('USER', 'unknown')
    
    cmd = ['squeue', '-u', user, '-h', '-o', '%i|%j|%T|%M|%N']
    rc, stdout, stderr = run_command(cmd)
    
    jobs = {}
    if rc == 0:
        for line in stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split('|')
            if len(parts) >= 5:
                job_id = parts[0].strip()
                jobs[job_id] = {
                    'name': parts[1].strip(),
                    'state': parts[2].strip(),
                    'time': parts[3].strip(),
                    'nodelist': parts[4].strip(),
                }
    return jobs


def backup_file(filepath: Path, backup_dir: Path = None) -> Path:
    """Create a timestamped backup of a file."""
    if not filepath.exists():
        return None
    
    if backup_dir is None:
        backup_dir = filepath.parent / "backups"
    
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{filepath.stem}_{timestamp}{filepath.suffix}"
    shutil.copy2(filepath, backup_path)
    
    return backup_path


# ============================================================================
# Priority System
# ============================================================================

def generate_priority_experiments(
    models: List[str],
    datasets: List[str],
    priority_config: Dict
) -> List[Tuple[ExperimentParams, int]]:
    """
    Generate experiments with priority scores.
    
    Priority levels (from config):
        1: Base (t=0.8, k=20, h=20, g=100)
        2: +k=1
        3: +k=v (vocab size)
        4: +t=0.1
        5: +t=1.0
        6: +k=10
        7: +t=0.3
        8: +k=5
        9: +h=10
        10: +g=50
    
    Returns:
        List of (ExperimentParams, priority) tuples, sorted by priority (descending)
    """
    # Default parameter values
    n = priority_config.get('n', 100)
    c = priority_config.get('c', 100)
    
    # Priority level definitions (level -> params to add)
    # Each level includes all previous levels
    priority_levels = {
        1: {'h': [20], 'g': [100], 'k': [20], 't': [0.8]},           # Base
        2: {'h': [20], 'g': [100], 'k': [1, 20], 't': [0.8]},        # +k=1
        3: {'h': [20], 'g': [100], 'k': [-1, 1, 20], 't': [0.8]},    # +k=v
        4: {'h': [20], 'g': [100], 'k': [-1, 1, 20], 't': [0.1, 0.8]},  # +t=0.1
        5: {'h': [20], 'g': [100], 'k': [-1, 1, 20], 't': [0.1, 0.8, 1.0]},  # +t=1.0
        6: {'h': [20], 'g': [100], 'k': [-1, 1, 10, 20], 't': [0.1, 0.8, 1.0]},  # +k=10
        7: {'h': [20], 'g': [100], 'k': [-1, 1, 10, 20], 't': [0.1, 0.3, 0.8, 1.0]},  # +t=0.3
        8: {'h': [20], 'g': [100], 'k': [-1, 1, 5, 10, 20], 't': [0.1, 0.3, 0.8, 1.0]},  # +k=5
        9: {'h': [10, 20], 'g': [100], 'k': [-1, 1, 5, 10, 20], 't': [0.1, 0.3, 0.8, 1.0]},  # +h=10
        10: {'h': [10, 20], 'g': [50, 100], 'k': [-1, 1, 5, 10, 20], 't': [0.1, 0.3, 0.8, 1.0]},  # +g=50
    }
    
    # Get max priority level from config
    max_priority = priority_config.get('max_priority_level', 10)
    
    # Generate all experiments up to max priority level
    all_experiments = []
    seen = set()
    
    for level in range(1, max_priority + 1):
        params = priority_levels.get(level, priority_levels[10])
        
        for model in models:
            for dataset in datasets:
                for h in params['h']:
                    for g in params['g']:
                        for k in params['k']:
                            for t in params['t']:
                                exp = ExperimentParams(
                                    n=n, c=c, h=h, g=g, k=k, t=t,
                                    model=model, dataset=dataset
                                )
                                exp_name = exp.experiment_name
                                
                                if exp_name not in seen:
                                    seen.add(exp_name)
                                    # Priority: higher level = added later = lower priority
                                    # So invert: priority = 1000 - level
                                    priority = 1000 - level
                                    all_experiments.append((exp, priority))
    
    # Sort by priority (descending)
    all_experiments.sort(key=lambda x: x[1], reverse=True)
    
    return all_experiments


# ============================================================================
# Completion Detection
# ============================================================================

def check_output_files(output_dir: Path, experiment: str, job_type: str) -> bool:
    """Check if output files exist for a job."""
    exp_dir = output_dir / experiment
    
    if job_type == 'gen':
        required = [
            'generated_sequences.parquet',
            'original_sequences.parquet',
            'generation_metadata.json',
        ]
    else:  # stats
        required = [
            'token_counts.csv',
            'statistics_metadata.json',
        ]
    
    return all((exp_dir / f).exists() for f in required)


def parse_log_for_completion(log_path: Path, job_type: str) -> Tuple[bool, Optional[str]]:
    """
    Parse a SLURM log file for completion markers.
    
    Uses COMPLETION_MARKERS dict which can be edited to add new patterns.
    """
    if not log_path.exists():
        return False, None
    
    try:
        content = log_path.read_text()
        
        # Check for completion markers
        markers = COMPLETION_MARKERS.get(job_type, COMPLETION_MARKERS.get('generation', []))
        
        for marker_set in markers:
            if all(m in content for m in marker_set):
                return True, None
        
        # Check for error patterns
        for pattern in ERROR_PATTERNS:
            match = re.search(pattern, content)
            if match:
                return False, match.group(1)[:200]  # Truncate error message
        
        return False, None
    
    except Exception as e:
        return False, f"Log parse error: {e}"


def find_log_file(log_dir: Path, job_name: str, job_id: str = None) -> Optional[Path]:
    """Find the log file for a job."""
    if job_id:
        # Try exact match first
        log_path = log_dir / f"{job_name}-{job_id}.out"
        if log_path.exists():
            return log_path
    
    # Try pattern match
    pattern = f"{job_name}-*.out"
    matches = list(log_dir.glob(pattern))
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime)
    
    # Try with experiment name only (job name format: NNN.exp_name)
    if '.' in job_name:
        exp_name = job_name.split('.', 1)[1]
        for prefix in ['gen', 'stats']:
            pattern = f"{prefix}*{exp_name}*.out"
            matches = list(log_dir.glob(pattern))
            if matches:
                return max(matches, key=lambda p: p.stat().st_mtime)
    
    return None


# ============================================================================
# SLURM Script Generation
# ============================================================================

def generate_slurm_script(
    exp: ExperimentParams,
    job_type: str,
    job_number: int,
    template_dir: Path,
    output_dir: Path,
    config_dir: Path,
    log_dir: Path,
) -> Path:
    """Generate a SLURM script for an experiment."""
    job_name = f"{job_number:03d}.{exp.experiment_name}"
    
    if job_type == 'gen':
        template = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH -p gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH -e {log_dir}/%x-%j.err
#SBATCH -o {log_dir}/%x-%j.out
#SBATCH --nodelist=PLACEHOLDER_NODE

echo "=========================================="
echo "Generation Job: {exp.experiment_name}"
echo "Model: {exp.model}"
echo "Dataset: {exp.dataset}"
echo "Parameters: n={exp.n}, c={exp.c}, h={exp.h}, g={exp.g}, k={exp.k}, t={exp.t}"
echo "=========================================="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

# Set GPU device
export CUDA_VISIBLE_DEVICES=PLACEHOLDER_GPU

# GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Load environment
cd ~/life-sequencing-dutch/
source requirements/load_venv.sh

# Run generation
python -m pop2vec.llm.src.new_code.gen_eval.src.generative_generate_sequences \\
    --config {config_dir}/{exp.experiment_name}/run_config.yaml

echo ""
echo "Generation Complete!"
echo "Completed: $(date)"
"""
    else:  # stats
        template = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH -p thin
#SBATCH -e {log_dir}/%x-%j.err
#SBATCH -o {log_dir}/%x-%j.out

echo "=========================================="
echo "Statistics + Plots Job: {exp.experiment_name}"
echo "Model: {exp.model}"
echo "Dataset: {exp.dataset}"
echo "=========================================="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

# Load environment
cd ~/life-sequencing-dutch/
source requirements/load_venv.sh

# Step 1: Statistics
echo ""
echo "=========================================="
echo "Step 1/2: Computing Statistics"
echo "=========================================="
python -m pop2vec.llm.src.new_code.gen_eval.src.generative_compute_statistics \\
    --config {config_dir}/{exp.experiment_name}/run_config.yaml

STATS_EXIT_CODE=$?
if [ $STATS_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Statistics computation failed with exit code $STATS_EXIT_CODE"
    exit $STATS_EXIT_CODE
fi
echo "Statistics Complete!"

# Step 2: Plots
echo ""
echo "=========================================="
echo "Step 2/2: Generating Plots"
echo "=========================================="
python -m pop2vec.llm.src.new_code.gen_eval.src.plot_statistics \\
    --config {config_dir}/{exp.experiment_name}/run_config.yaml

echo "Plots Complete!"
echo ""
echo "=========================================="
echo "All Steps Completed: $(date)"
echo "=========================================="
"""
    
    # Write script
    script_dir = template_dir
    script_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = 'gen' if job_type == 'gen' else 'stats'
    script_path = script_dir / f"{prefix}_{exp.experiment_name}.sh"
    script_path.write_text(template)
    script_path.chmod(0o755)
    
    return script_path


def patch_script_for_gpu(script_path: Path, node: str, gpu_index: int) -> Path:
    """Patch a SLURM script to use specific node and GPU."""
    content = script_path.read_text()
    
    # Replace placeholders or existing values
    content = re.sub(r'#SBATCH --nodelist=\S+', f'#SBATCH --nodelist={node}', content)
    content = re.sub(r'PLACEHOLDER_NODE', node, content)
    content = re.sub(r'export CUDA_VISIBLE_DEVICES=\S+', f'export CUDA_VISIBLE_DEVICES={gpu_index}', content)
    content = re.sub(r'PLACEHOLDER_GPU', str(gpu_index), content)
    
    # Write patched script
    patched_path = script_path.parent / f"{script_path.stem}.patched.sh"
    patched_path.write_text(content)
    patched_path.chmod(0o755)
    
    return patched_path


# ============================================================================
# Registry and Config Generation
# ============================================================================

def load_registry(registry_path: Path = None) -> Dict:
    """
    Load the dataset/model registry.
    
    The registry maps:
    - Dataset names (GD0, GDB0, etc.) -> H5 file paths
    - Model names (Gen-BASE, etc.) -> checkpoint path, vocab path
    - Model-dataset compatibility matrix
    """
    if registry_path is None:
        registry_path = DEFAULT_REGISTRY_PATH
    
    registry_path = Path(registry_path)
    
    if not registry_path.exists():
        logging.warning(f"[REGISTRY] Registry file not found: {registry_path}")
        return {'datasets': {}, 'models': {}, 'model_dataset_compatibility': {}}
    
    with open(registry_path, 'r') as f:
        registry = yaml.safe_load(f)
    
    logging.info(f"[REGISTRY] Loaded: {len(registry.get('datasets', {}))} datasets, "
                 f"{len(registry.get('models', {}))} models")
    
    return registry


def generate_experiment_config(
    exp: 'ExperimentParams',
    registry: Dict,
    config_dir: Path,
    output_dir: Path,
) -> Path:
    """
    Generate a per-experiment run_config.yaml file using the registry.
    
    Raises ValueError if dataset or model not found in registry, or paths are null.
    """
    exp_name = exp.experiment_name
    exp_config_dir = config_dir / exp_name
    exp_config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = exp_config_dir / "run_config.yaml"
    
    # Skip if config already exists
    if config_path.exists():
        return config_path
    
    # Look up dataset path
    datasets = registry.get('datasets', {})
    dataset_path = datasets.get(exp.dataset)
    if not dataset_path:
        raise ValueError(f"Dataset '{exp.dataset}' not found or null in registry")
    
    # Look up model info
    models = registry.get('models', {})
    model_info = models.get(exp.model)
    if not model_info:
        raise ValueError(f"Model '{exp.model}' not found in registry")
    
    checkpoint_path = model_info.get('checkpoint')
    if not checkpoint_path:
        raise ValueError(f"Checkpoint for '{exp.model}' is null in registry")
    
    vocab_path = model_info.get('vocab', registry.get('default_vocab'))
    
    # Experiment output directory
    exp_output_dir = output_dir / exp_name
    
    # Build config
    config = {
        'experiment_name': exp_name,
        'model_name': exp.model,
        'checkpoint_path': checkpoint_path,
        'vocab_path': vocab_path,
        'data_path': dataset_path,
        'output_dir': str(exp_output_dir),
        'num_people': exp.n,
        'num_generations': exp.c,
        'horizon': exp.h,
        'prefix_gap': exp.g,
        'top_k': exp.k if exp.k > 0 else None,
        'temperature': exp.t,
        'generation_batch_size': 64,
        'seed': 42,
        'sequences_path': str(exp_output_dir / 'generated_sequences.parquet'),
        'original_sequences_path': str(exp_output_dir / 'original_sequences.parquet'),
        'ages_path': str(exp_output_dir / 'ages.parquet'),
        'compute_by_age': True,
        'exclude_padding': False,
        'prefix_lengths': [7, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'statistics_path': str(exp_output_dir / f'statistics_n{exp.n}_c{exp.c}_full.csv'),
        'statistics_summary_path': str(exp_output_dir / f'statistics_n{exp.n}_c{exp.c}_summary.csv'),
        'statistics_by_age_path': str(exp_output_dir / f'statistics_by_age_n{exp.n}_c{exp.c}_full.csv'),
        'statistics_by_age_summary_path': str(exp_output_dir / f'statistics_by_age_n{exp.n}_c{exp.c}_summary.csv'),
        'generation': {'partition': 'gpu_h100', 'gpus': 1, 'cpus': 4, 'mem': '64G', 'time': '48:00:00'},
        'statistics': {'partition': 'thin', 'cpus': 8, 'mem': '32G', 'time': '12:00:00'},
    }
    
    if exp.k == -1:
        config['use_full_vocab'] = True
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=True)
    
    logging.info(f"[CONFIG] Generated: {config_path}")
    return config_path


def validate_registry(registry: Dict, models: List[str], datasets: List[str]) -> List[str]:
    """Validate that all required models and datasets have valid paths in registry."""
    errors = []
    
    reg_datasets = registry.get('datasets', {})
    for ds in datasets:
        if ds not in reg_datasets or not reg_datasets[ds]:
            errors.append(f"Dataset '{ds}' missing or null in registry")
    
    reg_models = registry.get('models', {})
    for model in models:
        if model not in reg_models or not reg_models[model]:
            errors.append(f"Model '{model}' missing or null in registry")
        elif not reg_models[model].get('checkpoint'):
            errors.append(f"Model '{model}' checkpoint is null in registry")
    
    return errors


def get_compatible_datasets(model: str, registry: Dict, all_datasets: List[str]) -> List[str]:
    """Get datasets compatible with a model according to registry."""
    compatibility = registry.get('model_dataset_compatibility', {})
    if not compatibility:
        return all_datasets
    compatible = compatibility.get(model, all_datasets)
    return [ds for ds in compatible if ds in all_datasets]


def submit_job(script_path: str, node: str = None, gpu_index: int = None) -> Optional[str]:
    """Submit a SLURM job."""
    script_path = Path(script_path)
    
    # Patch script if node/gpu specified
    if node is not None and gpu_index is not None:
        script_path = patch_script_for_gpu(script_path, node, gpu_index)
    
    cmd = ['sbatch', str(script_path)]
    rc, stdout, stderr = run_command(cmd)
    
    if rc == 0:
        match = re.search(r'Submitted batch job (\d+)', stdout)
        if match:
            return match.group(1)
    
    logging.error(f"Failed to submit {script_path}: {stderr}")
    return None


# ============================================================================
# Result Aggregation
# ============================================================================

def aggregate_results(
    output_dir: Path,
    experiments: List[str],
    export_dir: Path
) -> Dict[str, Path]:
    """
    Aggregate results from all completed experiments into CSVs.
    
    Creates:
        1. statistics_block_summary.csv - Block-wise comparison statistics
        2. statistics_decade_summary.csv - Decade-wise token frequencies
        3. token_counts_merged.csv - Merged token counts with experiment metadata
    
    Each row includes all experiment parameters as columns.
    """
    import pandas as pd
    
    export_dir.mkdir(parents=True, exist_ok=True)
    
    block_rows = []
    decade_rows = []
    token_rows = []
    
    for exp_name in experiments:
        exp_dir = output_dir / exp_name
        
        if not exp_dir.exists():
            continue
        
        # Parse experiment parameters
        try:
            params = ExperimentParams.from_name(exp_name)
            params_dict = params.to_dict()
        except Exception as e:
            logging.warning(f"Cannot parse experiment name {exp_name}: {e}")
            params_dict = {'experiment': exp_name}
        
        # Read block-wise summary
        block_path = exp_dir / 'statistics_summary.csv'
        if block_path.exists():
            try:
                df = pd.read_csv(block_path)
                # Add experiment params as columns
                for key, val in params_dict.items():
                    df[key] = val
                block_rows.append(df)
            except Exception as e:
                logging.warning(f"Cannot read {block_path}: {e}")
        
        # Read decade-wise summary (token frequencies by age)
        decade_path = exp_dir / 'statistics_by_age_summary.csv'
        if decade_path.exists():
            try:
                df = pd.read_csv(decade_path)
                # Drop row_type column if all values are 'token_frequency'
                if 'row_type' in df.columns:
                    if df['row_type'].nunique() == 1:
                        df = df.drop(columns=['row_type'])
                # Add experiment params
                for key, val in params_dict.items():
                    df[key] = val
                decade_rows.append(df)
            except Exception as e:
                logging.warning(f"Cannot read {decade_path}: {e}")
        
        # Read token counts
        token_path = exp_dir / 'token_counts.csv'
        if token_path.exists():
            try:
                df = pd.read_csv(token_path)
                for key, val in params_dict.items():
                    df[key] = val
                token_rows.append(df)
            except Exception as e:
                logging.warning(f"Cannot read {token_path}: {e}")
    
    results = {}
    
    # Merge and save block summary
    if block_rows:
        merged_block = pd.concat(block_rows, ignore_index=True)
        # Drop all-zero rows (where all numeric columns are 0)
        numeric_cols = merged_block.select_dtypes(include=['number']).columns
        non_zero_mask = (merged_block[numeric_cols] != 0).any(axis=1)
        merged_block = merged_block[non_zero_mask]
        # Replace 0 with empty string for sparsity
        for col in numeric_cols:
            merged_block[col] = merged_block[col].replace(0, '')
        
        block_out = export_dir / 'statistics_block_summary.csv'
        merged_block.to_csv(block_out, index=False)
        results['block_summary'] = block_out
        logging.info(f"Saved block summary: {block_out} ({len(merged_block)} rows)")
    
    # Merge and save decade summary
    if decade_rows:
        merged_decade = pd.concat(decade_rows, ignore_index=True)
        numeric_cols = merged_decade.select_dtypes(include=['number']).columns
        non_zero_mask = (merged_decade[numeric_cols] != 0).any(axis=1)
        merged_decade = merged_decade[non_zero_mask]
        for col in numeric_cols:
            merged_decade[col] = merged_decade[col].replace(0, '')
        
        decade_out = export_dir / 'statistics_decade_summary.csv'
        merged_decade.to_csv(decade_out, index=False)
        results['decade_summary'] = decade_out
        logging.info(f"Saved decade summary: {decade_out} ({len(merged_decade)} rows)")
    
    # Merge and save token counts
    if token_rows:
        merged_tokens = pd.concat(token_rows, ignore_index=True)
        numeric_cols = merged_tokens.select_dtypes(include=['number']).columns
        non_zero_mask = (merged_tokens[numeric_cols] != 0).any(axis=1)
        merged_tokens = merged_tokens[non_zero_mask]
        for col in numeric_cols:
            merged_tokens[col] = merged_tokens[col].replace(0, '')
        
        token_out = export_dir / 'token_counts_merged.csv'
        merged_tokens.to_csv(token_out, index=False)
        results['token_counts'] = token_out
        logging.info(f"Saved token counts: {token_out} ({len(merged_tokens)} rows)")
    
    return results


# ============================================================================
# Supervisor Core
# ============================================================================

class SupervisorV2:
    """Enhanced pipeline supervisor with runtime config updates."""
    
    def __init__(
        self,
        config_path: Path,
        slurm_dir: Path,
        output_dir: Path,
        log_dir: Path,
        state_dir: Path,
        export_dir: Path,
        config_dir: Path = None,
        registry_path: Path = None,
    ):
        self.config_path = Path(config_path)
        self.slurm_dir = Path(slurm_dir)
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.state_dir = Path(state_dir)
        self.export_dir = Path(export_dir)
        self.config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
        self.registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
        
        # Create directories
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.slurm_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup supervisor script
        self._backup_self()
        
        # Load config
        self.config = self._load_config()
        self.config_mtime = self.config_path.stat().st_mtime
        
        # Load registry for dataset/model paths
        self.registry = load_registry(self.registry_path)
        self._validate_registry_on_startup()
        
        # State files
        self.state_file = self.state_dir / "pipeline_state_v2.json"
        self.dashboard_file = self.state_dir / "dashboard_v2.txt"
        self.gpu_config_file = self.state_dir / "gpu_config.yaml"
        self.priority_file = self.state_dir / "priorities.yaml"
        
        # Load or create state
        self.state = self._load_or_create_state()
        
        # Runtime GPU config (can be modified while running)
        self.gpus = self._load_gpu_config()
        self.gpu_slots = self._build_gpu_slots()
        
        logging.info(f"[SUPERVISOR V2] Initialized")
        logging.info(f"  Config: {self.config_path}")
        logging.info(f"  Registry: {self.registry_path}")
        logging.info(f"  Config Dir: {self.config_dir}")
        logging.info(f"  Experiments: {len(self.state.experiments)}")
        logging.info(f"  GPU Slots: {len(self.gpu_slots)}")
    
    def _validate_registry_on_startup(self):
        """Validate registry has paths for all configured models and datasets."""
        models = self.config.get('models', [])
        datasets = self.config.get('datasets', [])
        
        errors = validate_registry(self.registry, models, datasets)
        if errors:
            logging.warning("[REGISTRY] Validation warnings:")
            for err in errors:
                logging.warning(f"  - {err}")
            logging.warning("  Fill in paths in registry.yaml before jobs can run")
    
    def _backup_self(self):
        """Create a backup of the supervisor script."""
        script_path = Path(__file__)
        backup_dir = self.state_dir / "backups"
        backup_path = backup_file(script_path, backup_dir)
        if backup_path:
            logging.info(f"[BACKUP] Supervisor backed up to: {backup_path}")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _check_config_update(self) -> bool:
        """Check if config file has been modified."""
        try:
            current_mtime = self.config_path.stat().st_mtime
            if current_mtime > self.config_mtime:
                logging.info("[CONFIG] Configuration file changed, reloading...")
                self.config = self._load_config()
                self.config_mtime = current_mtime
                self.state.config_version += 1
                return True
        except Exception as e:
            logging.warning(f"[CONFIG] Error checking config: {e}")
        return False
    
    def _load_gpu_config(self) -> Dict[str, List[int]]:
        """
        Load GPU configuration.
        
        Checks for runtime GPU config file first, then falls back to main config.
        This allows runtime modification of available GPUs.
        """
        # Check for runtime GPU config
        if self.gpu_config_file.exists():
            try:
                with open(self.gpu_config_file, 'r') as f:
                    gpu_config = yaml.safe_load(f)
                logging.info(f"[GPU] Loaded runtime GPU config from {self.gpu_config_file}")
                return gpu_config.get('gpus', {})
            except Exception as e:
                logging.warning(f"[GPU] Error loading runtime GPU config: {e}")
        
        # Fall back to main config
        return self.config.get('gpus', {})
    
    def _build_gpu_slots(self) -> List[str]:
        """Build list of GPU slots from config."""
        slots = []
        for node, indices in sorted(self.gpus.items()):
            for idx in sorted(indices):
                slots.append(f"{node}:{idx}")
        return slots
    
    def _load_or_create_state(self) -> PipelineState:
        """Load existing state or create new one."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                state = PipelineState.from_dict(data)
                logging.info(f"[STATE] Loaded existing state with {len(state.jobs)} jobs")
                return state
            except Exception as e:
                logging.warning(f"[STATE] Error loading state: {e}, creating new")
        
        # Create new state
        state = PipelineState(start_time=datetime.now().isoformat())
        
        # Generate experiments based on priority config
        priority_config = self.config.get('priority', {})
        models = self.config.get('models', [])
        datasets = self.config.get('datasets', [])
        
        experiments_with_priority = generate_priority_experiments(
            models, datasets, priority_config
        )
        
        # Initialize jobs
        for exp, priority in experiments_with_priority:
            exp_name = exp.experiment_name
            state.experiments.append(exp_name)
            
            # Generation job
            state.job_counter += 1
            gen_job = JobState(
                experiment=exp_name,
                job_type='gen',
                job_number=state.job_counter,
                priority=priority,
            )
            state.jobs[gen_job.key] = gen_job
            
            # Statistics job
            state.job_counter += 1
            stats_job = JobState(
                experiment=exp_name,
                job_type='stats',
                job_number=state.job_counter,
                priority=priority,
            )
            state.jobs[stats_job.key] = stats_job
        
        logging.info(f"[STATE] Created new state with {len(state.jobs)} jobs")
        return state
    
    def _save_state(self):
        """Save current state to file."""
        self.state.last_update = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
    
    def _get_available_gpu_slots(self) -> List[str]:
        """Get GPU slots not currently in use."""
        used_slots = set()
        for job in self.state.jobs.values():
            if job.gpu_slot and job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
                used_slots.add(job.gpu_slot)
        
        return [s for s in self.gpu_slots if s not in used_slots]
    
    def _update_jobs_from_slurm(self):
        """Update job states from SLURM queue."""
        slurm_jobs = get_slurm_jobs()
        slurm_job_names = {v['name']: k for k, v in slurm_jobs.items()}
        
        for job in self.state.jobs.values():
            if job.status not in [JobStatus.QUEUED, JobStatus.RUNNING]:
                continue
            
            # Check if job is still in queue
            if job.slurm_job_id and job.slurm_job_id in slurm_jobs:
                slurm_state = slurm_jobs[job.slurm_job_id]['state']
                if slurm_state == 'RUNNING' and job.status == JobStatus.QUEUED:
                    job.status = JobStatus.RUNNING
                    job.start_time = datetime.now().isoformat()
            else:
                # Job disappeared from queue - check completion
                self._check_job_completion(job)
    
    def _check_job_completion(self, job: JobState):
        """Check if a job completed successfully."""
        # Check output files
        if check_output_files(self.output_dir, job.experiment, job.job_type):
            job.status = JobStatus.COMPLETED
            job.end_time = datetime.now().isoformat()
            if job.start_time:
                start = datetime.fromisoformat(job.start_time)
                end = datetime.fromisoformat(job.end_time)
                job.duration_seconds = (end - start).total_seconds()
            if job.gpu_slot:
                self.state.gpu_assignments.pop(job.gpu_slot, None)
            return
        
        # Check log files
        log_file = find_log_file(self.log_dir, job.job_name, job.slurm_job_id)
        if log_file:
            is_complete, error = parse_log_for_completion(log_file, job.job_type)
            if is_complete:
                job.status = JobStatus.COMPLETED
                job.end_time = datetime.now().isoformat()
                if job.gpu_slot:
                    self.state.gpu_assignments.pop(job.gpu_slot, None)
                return
            if error:
                job.status = JobStatus.FAILED
                job.error_message = error
                job.end_time = datetime.now().isoformat()
                if job.gpu_slot:
                    self.state.gpu_assignments.pop(job.gpu_slot, None)
                return
        
        # Cannot determine status - mark as failed
        job.status = JobStatus.FAILED
        job.error_message = "Job disappeared from queue without completion markers"
        job.end_time = datetime.now().isoformat()
        if job.gpu_slot:
            self.state.gpu_assignments.pop(job.gpu_slot, None)
    
    def _submit_generation_jobs(self):
        """Submit generation jobs for available GPU slots."""
        available_slots = self._get_available_gpu_slots()
        if not available_slots:
            return
        
        # Get pending jobs sorted by priority
        pending_jobs = []
        for job in self.state.jobs.values():
            if job.job_type != 'gen':
                continue
            if job.status == JobStatus.NOT_STARTED:
                pending_jobs.append(job)
            elif job.status == JobStatus.FAILED:
                if self.config.get('auto_resubmit_failed', False):
                    if job.retry_count < self.config.get('max_retries', 2):
                        pending_jobs.append(job)
        
        # Sort by priority (descending)
        pending_jobs.sort(key=lambda j: j.priority, reverse=True)
        
        # Check min_priority from priority config (not top-level config)
        priority_config = self.config.get('priority', {})
        min_priority = priority_config.get('min_priority', 0)
        
        # Submit jobs
        for slot in available_slots:
            if not pending_jobs:
                break
            
            job = pending_jobs.pop(0)
            
            # Check priority threshold
            if job.priority < min_priority:
                continue
            
            try:
                exp = ExperimentParams.from_name(job.experiment)
                
                # Generate per-experiment config using registry
                try:
                    generate_experiment_config(
                        exp, self.registry, self.config_dir, self.output_dir
                    )
                except ValueError as e:
                    logging.error(f"[CONFIG] Cannot generate config for {job.experiment}: {e}")
                    continue
                
                # Get or generate SLURM script
                script_path = self.slurm_dir / f"gen_{job.experiment}.sh"
                if not script_path.exists():
                    script_path = generate_slurm_script(
                        exp, 'gen', job.job_number,
                        self.slurm_dir, self.output_dir,
                        self.config_dir, self.log_dir
                    )
            except Exception as e:
                logging.error(f"Cannot setup job {job.experiment}: {e}")
                continue
            
            # Parse slot
            node, gpu_idx = slot.split(':')
            gpu_idx = int(gpu_idx)
            
            # Submit
            slurm_id = submit_job(str(script_path), node, gpu_idx)
            
            if slurm_id:
                job.slurm_job_id = slurm_id
                job.status = JobStatus.QUEUED
                job.gpu_slot = slot
                job.submit_time = datetime.now().isoformat()
                if job.retry_count > 0:
                    job.retry_count += 1
                self.state.gpu_assignments[slot] = job.key
                logging.info(f"[SUBMIT] {job.job_name} -> {slot} (SLURM ID: {slurm_id})")
            else:
                logging.error(f"[ERROR] Failed to submit {job.experiment}")
    
    def _submit_statistics_jobs(self):
        """Submit statistics jobs for completed generation jobs."""
        for job in self.state.jobs.values():
            if job.job_type != 'stats':
                continue
            if job.status != JobStatus.NOT_STARTED:
                continue
            
            # Check if generation is complete
            gen_key = f"gen:{job.experiment}"
            gen_job = self.state.jobs.get(gen_key)
            
            if not gen_job or gen_job.status != JobStatus.COMPLETED:
                continue
            
            # Get or generate script (config should already exist from generation)
            script_path = self.slurm_dir / f"stats_{job.experiment}.sh"
            if not script_path.exists():
                try:
                    exp = ExperimentParams.from_name(job.experiment)
                    script_path = generate_slurm_script(
                        exp, 'stats', job.job_number,
                        self.slurm_dir, self.output_dir,
                        self.config_dir, self.log_dir
                    )
                except Exception as e:
                    logging.error(f"Cannot generate stats script for {job.experiment}: {e}")
                    continue
            
            # Submit (CPU job, no GPU slot needed)
            slurm_id = submit_job(str(script_path))
            
            if slurm_id:
                job.slurm_job_id = slurm_id
                job.status = JobStatus.QUEUED
                job.submit_time = datetime.now().isoformat()
                logging.info(f"[SUBMIT] Stats {job.job_name} (SLURM ID: {slurm_id})")
    
    def _write_dashboard(self):
        """Write human-readable dashboard."""
        lines = []
        lines.append("=" * 70)
        lines.append("PIPELINE SUPERVISOR V2 DASHBOARD")
        lines.append(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        
        # Summary
        status_counts = {}
        for job in self.state.jobs.values():
            status_counts[job.status.value] = status_counts.get(job.status.value, 0) + 1
        
        lines.append("\nJOB STATUS SUMMARY:")
        for status, count in sorted(status_counts.items()):
            lines.append(f"  {status}: {count}")
        
        lines.append(f"\nGPU SLOTS: {len(self.gpu_slots)}")
        lines.append(f"  Available: {len(self._get_available_gpu_slots())}")
        lines.append(f"  In use: {len(self.state.gpu_assignments)}")
        
        # GPU assignments
        if self.state.gpu_assignments:
            lines.append("\nCURRENT GPU ASSIGNMENTS:")
            for slot, job_key in self.state.gpu_assignments.items():
                job = self.state.jobs.get(job_key)
                if job:
                    lines.append(f"  {slot}: {job.experiment} ({job.status.value})")
        
        # Recent activity
        lines.append("\nRECENT JOBS:")
        recent_jobs = sorted(
            [j for j in self.state.jobs.values() if j.submit_time],
            key=lambda j: j.submit_time or '',
            reverse=True
        )[:10]
        
        for job in recent_jobs:
            status_icon = {
                'completed': '✓',
                'running': '▶',
                'queued': '⏳',
                'failed': '✗',
            }.get(job.status.value, '?')
            lines.append(f"  {status_icon} {job.job_name[:50]} - {job.status.value}")
        
        # Instructions
        lines.append("\n" + "=" * 70)
        lines.append("RUNTIME CONFIGURATION:")
        lines.append(f"  Edit GPU config: {self.gpu_config_file}")
        lines.append(f"  Edit priorities: {self.priority_file}")
        lines.append("  Changes are detected automatically")
        lines.append("=" * 70)
        
        self.dashboard_file.write_text('\n'.join(lines))
    
    def _reload_runtime_configs(self):
        """Reload runtime configurations (GPUs, priorities)."""
        # Reload GPU config
        new_gpus = self._load_gpu_config()
        if new_gpus != self.gpus:
            old_slots = set(self.gpu_slots)
            self.gpus = new_gpus
            self.gpu_slots = self._build_gpu_slots()
            new_slots = set(self.gpu_slots)
            
            added = new_slots - old_slots
            removed = old_slots - new_slots
            
            if added:
                logging.info(f"[GPU] Added slots: {added}")
            if removed:
                logging.info(f"[GPU] Removed slots: {removed}")
                # Clean up removed slot assignments
                for slot in removed:
                    if slot in self.state.gpu_assignments:
                        del self.state.gpu_assignments[slot]
    
    def run_once(self):
        """Run one iteration of the supervisor loop."""
        # Check for config updates
        self._check_config_update()
        self._reload_runtime_configs()
        
        # Update job states from SLURM
        self._update_jobs_from_slurm()
        
        # Submit new jobs
        self._submit_generation_jobs()
        
        if self.config.get('auto_submit_stats', True):
            self._submit_statistics_jobs()
        
        # Save state and dashboard
        self._save_state()
        self._write_dashboard()
        
        # Aggregate results periodically
        if self.config.get('auto_aggregate', True):
            completed_exps = [
                j.experiment for j in self.state.jobs.values()
                if j.job_type == 'stats' and j.status == JobStatus.COMPLETED
            ]
            if completed_exps:
                try:
                    aggregate_results(self.output_dir, completed_exps, self.export_dir)
                except Exception as e:
                    logging.warning(f"[AGGREGATE] Error: {e}")
    
    def run(self, max_iterations: int = None):
        """Run the supervisor loop."""
        iteration = 0
        poll_interval = self.config.get('poll_interval_seconds', 60)
        
        logging.info(f"[SUPERVISOR] Starting main loop (poll interval: {poll_interval}s)")
        
        try:
            while True:
                iteration += 1
                logging.info(f"\n[ITERATION {iteration}] {datetime.now().strftime('%H:%M:%S')}")
                
                self.run_once()
                
                # Check if all jobs are done
                pending = sum(1 for j in self.state.jobs.values() 
                             if j.status in [JobStatus.NOT_STARTED, JobStatus.QUEUED, JobStatus.RUNNING])
                
                if pending == 0:
                    logging.info("[SUPERVISOR] All jobs completed!")
                    break
                
                if max_iterations and iteration >= max_iterations:
                    logging.info(f"[SUPERVISOR] Reached max iterations ({max_iterations})")
                    break
                
                time.sleep(poll_interval)
        
        except KeyboardInterrupt:
            logging.info("\n[SUPERVISOR] Interrupted by user")
        
        finally:
            self._save_state()
            self._write_dashboard()
            logging.info("[SUPERVISOR] Shutdown complete")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Supervisor v2 for Generative Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Runtime Configuration:
    The supervisor monitors these files for runtime changes:
    
    GPU Config (supervisor_state/gpu_config.yaml):
        gpus:
          ossc9424vm1: [0, 1, 2, 3]
          ossc9424vm2: [0, 1]
    
    To add/remove GPUs at runtime, edit this file.
    Changes are detected automatically.

Completion Markers:
    To add new completion markers, edit the COMPLETION_MARKERS dict
    in supervisor_v2.py.

Examples:
    # Run with config file
    python supervisor_v2.py --config supervisor_v2_config.yaml
    
    # Run with custom paths
    python supervisor_v2.py --config config.yaml --output-dir /path/to/output
        """
    )
    
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--slurm-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--state-dir", type=str, default=None)
    parser.add_argument("--export-dir", type=str, default=None)
    parser.add_argument("--config-dir", type=str, default=None,
                        help="Directory for per-experiment run_config.yaml files")
    parser.add_argument("--registry", type=str, default=None,
                        help="Path to registry.yaml with dataset/model paths")
    parser.add_argument("--max-iterations", type=int, help="Max iterations (for testing)")
    
    args = parser.parse_args()
    
    # Load config to get paths (config file paths take precedence over defaults)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Priority: CLI arg > config file > default
    def resolve_path(cli_val, config_key, default_val):
        if cli_val is not None:
            return cli_val
        if config_key in config and config[config_key]:
            return config[config_key]
        return str(default_val)
    
    supervisor = SupervisorV2(
        config_path=args.config,
        slurm_dir=resolve_path(args.slurm_dir, 'slurm_dir', DEFAULT_SLURM_DIR),
        output_dir=resolve_path(args.output_dir, 'output_dir', DEFAULT_OUTPUT_DIR),
        log_dir=resolve_path(args.log_dir, 'log_dir', DEFAULT_LOG_DIR),
        state_dir=resolve_path(args.state_dir, 'state_dir', DEFAULT_STATE_DIR),
        export_dir=resolve_path(args.export_dir, 'export_dir', DEFAULT_EXPORT_DIR),
        config_dir=resolve_path(args.config_dir, 'config_dir', DEFAULT_CONFIG_DIR),
        registry_path=resolve_path(args.registry, 'registry_path', DEFAULT_REGISTRY_PATH),
    )
    
    supervisor.run(max_iterations=args.max_iterations)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    main()
