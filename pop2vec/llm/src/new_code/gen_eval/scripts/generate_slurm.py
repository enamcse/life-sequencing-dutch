#!/usr/bin/env python3
"""
Generate SLURM Scripts for Generative Evaluation

Creates SLURM submission scripts for specified models and experiment configurations.

Usage:
    # With experiment config file (multiple experiments)
    python generate_slurm.py --config experiments_config.yaml

    # With experiment name
    python generate_slurm.py --models model_v1 model_v2 --experiment exp_n10_c100

    # With direct parameters
    python generate_slurm.py --models model_v1 --n 10 --c 100 --h 20 --g 100

Config file format (YAML):
    models:
      - model_v1
      - model_v2
    experiments:
      - name: exp_n10_c100_h20_g100
        n: 10
        c: 100
        h: 20
        g: 100
      - name: exp_n100_c1000_h20_g100
        n: 100
        c: 1000
        h: 20
        g: 100
"""

import argparse
import os
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass


# Template for SLURM generation script (GPU)
SLURM_GENERATE_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name=gen_{model_name}_{exp_name}
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH -p {partition}
#SBATCH --gpus-per-node={gpus}
#SBATCH -e {log_dir}/%x-%j.err
#SBATCH -o {log_dir}/%x-%j.out
{nodelist_line}
{dependency_line}

echo "=========================================="
echo "Generation Job: {model_name}"
echo "Experiment: {exp_name}"
echo "GPU Index: {gpu_index}"
echo "Node: {node_name}"
echo "=========================================="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

# Set GPU device
{cuda_device_line}

# GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Load environment
cd ~/life-sequencing-dutch/
source requirements/load_venv.sh

# Run generation
python -m pop2vec.llm.src.new_code.gen_eval.src.generate_sequences \\
    --config {config_path}

echo ""
echo "Completed: $(date)"
'''

# Template for SLURM statistics script (CPU)
# This script runs both statistics computation AND plot generation
SLURM_STATS_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name=stats_{model_name}_{exp_name}
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH -p {partition}
#SBATCH -e {log_dir}/%x-%j.err
#SBATCH -o {log_dir}/%x-%j.out

echo "=========================================="
echo "Statistics + Plots Job: {model_name}"
echo "Experiment: {exp_name}"
echo "=========================================="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

# Load environment
cd ~/life-sequencing-dutch/
source requirements/load_venv.sh

# Step 1: Run statistics computation
echo ""
echo "=========================================="
echo "Step 1/2: Computing Statistics"
echo "=========================================="
python -m pop2vec.llm.src.new_code.gen_eval.src.compute_statistics \\
    --config {config_path}

STATS_EXIT_CODE=$?
if [ $STATS_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Statistics computation failed with exit code $STATS_EXIT_CODE"
    exit $STATS_EXIT_CODE
fi

echo ""
echo "Statistics Complete!"

# Step 2: Generate plots
echo ""
echo "=========================================="
echo "Step 2/2: Generating Plots"
echo "=========================================="
python -m pop2vec.llm.src.new_code.gen_eval.src.plot_statistics \\
    --config {config_path}

PLOT_EXIT_CODE=$?
if [ $PLOT_EXIT_CODE -ne 0 ]; then
    echo "WARNING: Plot generation failed with exit code $PLOT_EXIT_CODE"
    echo "Statistics were computed successfully, but plots failed."
    # Don't exit with error - stats are more important
fi

echo ""
echo "Plots Complete!"
echo ""
echo "=========================================="
echo "All Steps Completed: $(date)"
echo "=========================================="
'''


def load_model_config(model_name: str, base_dir: Path) -> dict:
    """Load model configuration from folder."""
    config_path = base_dir / model_name / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_experiment_config(exp_name: str, base_dir: Path) -> dict:
    """Load experiment configuration."""
    config_path = base_dir / f"{exp_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_experiment_config(n: int, c: int, h: int, g: int, name: str = None, 
                             exclude_padding: bool = True,
                             generation_batch_size: int = 64,
                             compute_by_age: bool = False,
                             prefix_lengths: List[int] = None) -> dict:
    """Create experiment configuration from parameters."""
    if name is None:
        name = f"exp_n{n}_c{c}_h{h}_g{g}"
    
    # Use explicit prefix_lengths if provided, otherwise generate from g
    if prefix_lengths is None:
        # Generate prefix lengths: 7, 100, 200, 300, ..., 1000
        prefix_lengths = [7]
        current = 100
        while current <= 1000:
            prefix_lengths.append(current)
            current += g
    
    return {
        'experiment_name': name,
        'num_people': n,
        'num_generations': c,
        'horizon': h,
        'prefix_gap': g,
        'prefix_lengths': prefix_lengths,
        'exclude_padding': exclude_padding,
        'generation_batch_size': generation_batch_size,
        'compute_by_age': compute_by_age,
        'top_k': 20,
        'temperature': 1.0,
        'seed': 42,
        # SLURM settings
        'generation': {
            'partition': 'gpu_h100',
            'gpus': 1,
            'cpus': 4,
            'mem': '64G',
            'time': '48:00:00',
        },
        'statistics': {
            'partition': 'thin',
            'cpus': 8,
            'mem': '32G',
            'time': '12:00:00',
        }
    }


def generate_run_config(model_config: dict, exp_config: dict, output_dir: Path) -> Path:
    """Generate combined config for a run."""
    n = exp_config.get('num_people', 10)
    c = exp_config.get('num_generations', 100)
    
    # Get the standard events config path
    script_dir = Path(__file__).parent.parent
    events_config_path = script_dir / 'config' / 'events_config.yaml'
    
    run_config = {
        **exp_config,
        'model_name': model_config['model_name'],
        'checkpoint_path': model_config['checkpoint_path'],
        'data_path': model_config['data_path'],
        'vocab_path': model_config['vocab_path'],
        'output_dir': str(output_dir),
        'sequences_path': str(output_dir / 'generated_sequences.parquet'),
        'original_sequences_path': str(output_dir / 'original_sequences.parquet'),
        'ages_path': str(output_dir / 'ages.parquet'),
        # Events config for plotting
        'events_config_path': str(events_config_path),
        # Include n and c in statistics filenames
        'statistics_path': str(output_dir / f'statistics_n{n}_c{c}_full.csv'),
        'statistics_summary_path': str(output_dir / f'statistics_n{n}_c{c}_summary.csv'),
        # By-age statistics paths
        'statistics_by_age_path': str(output_dir / f'statistics_by_age_n{n}_c{c}_full.csv'),
        'statistics_by_age_summary_path': str(output_dir / f'statistics_by_age_n{n}_c{c}_summary.csv'),
    }
    
    config_path = output_dir / 'run_config.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(run_config, f, default_flow_style=False)
    
    return config_path


def generate_slurm_scripts(
    models: list,
    exp_config: dict,
    models_base_dir: Path,
    output_base_dir: Path,
    slurm_output_dir: Path,
    log_dir: str,
    gpu_assignment: Dict[str, Dict[str, 'GpuSlot']] = None,
):
    """Generate SLURM scripts for all models.
    
    Args:
        models: List of model names
        exp_config: Experiment configuration
        models_base_dir: Base directory for model configs
        output_base_dir: Base directory for outputs
        slurm_output_dir: Directory for SLURM scripts
        log_dir: Directory for SLURM logs
        gpu_assignment: Dict[exp_name][model_name] = GpuSlot(node, gpu_index)
    
    Returns:
        list: scripts info list
    """
    exp_name = exp_config['experiment_name']
    gen_settings = exp_config.get('generation', {})
    stats_settings = exp_config.get('statistics', {})
    
    slurm_output_dir.mkdir(parents=True, exist_ok=True)
    
    scripts = []
    
    for model_name in models:
        # Load model config
        model_config = load_model_config(model_name, models_base_dir)
        
        # Create output directory
        run_output_dir = output_base_dir / model_name / exp_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run config
        config_path = generate_run_config(model_config, exp_config, run_output_dir)
        
        # Get GPU assignment for this model in this experiment
        gpu_slot = None
        gpu_index = -1
        node_name = ""
        nodelist_line = ""
        cuda_device_line = "# No GPU assignment"
        
        if gpu_assignment and exp_name in gpu_assignment:
            gpu_slot = gpu_assignment[exp_name].get(model_name)
            if gpu_slot:
                gpu_index = gpu_slot.gpu_index
                node_name = gpu_slot.node
                nodelist_line = f"#SBATCH --nodelist={node_name}"
                cuda_device_line = f"export CUDA_VISIBLE_DEVICES={gpu_index}"
        
        # Generate SLURM script for generation (GPU)
        # Note: dependency_line is empty here; dependencies are added at submit time
        gen_script_path = slurm_output_dir / f"gen_{model_name}_{exp_name}.sh"
        gen_script = SLURM_GENERATE_TEMPLATE.format(
            model_name=model_name,
            exp_name=exp_name,
            partition=gen_settings.get('partition', 'gpu_h100'),
            gpus=gen_settings.get('gpus', 1),
            cpus=gen_settings.get('cpus', 4),
            mem=gen_settings.get('mem', '64G'),
            time=gen_settings.get('time', '48:00:00'),
            log_dir=log_dir,
            config_path=config_path,
            dependency_line="",  # Dependencies added at submit time
            gpu_index=gpu_index,
            node_name=node_name,
            nodelist_line=nodelist_line,
            cuda_device_line=cuda_device_line,
        )
        
        with open(gen_script_path, 'w') as f:
            f.write(gen_script)
        os.chmod(gen_script_path, 0o755)
        
        # Generate SLURM script for statistics (CPU)
        stats_script_path = slurm_output_dir / f"stats_{model_name}_{exp_name}.sh"
        stats_script = SLURM_STATS_TEMPLATE.format(
            model_name=model_name,
            exp_name=exp_name,
            partition=stats_settings.get('partition', 'thin'),
            cpus=stats_settings.get('cpus', 8),
            mem=stats_settings.get('mem', '32G'),
            time=stats_settings.get('time', '12:00:00'),
            log_dir=log_dir,
            config_path=config_path,
        )
        
        with open(stats_script_path, 'w') as f:
            f.write(stats_script)
        os.chmod(stats_script_path, 0o755)
        
        scripts.append({
            'model': model_name,
            'experiment': exp_name,
            'gen_script': str(gen_script_path),
            'stats_script': str(stats_script_path),
            'config': str(config_path),
            'output_dir': str(run_output_dir),
            'gpu_index': gpu_index,
            'node': node_name,
        })
        
        print(f"  ✓ {model_name}")
        print(f"      Config: {config_path}")
        print(f"      Gen script: {gen_script_path}")
        print(f"      Stats script: {stats_script_path}")
        if gpu_slot:
            print(f"      GPU: {node_name}:GPU{gpu_index}")
        print()
    
    # Write manifest
    manifest_path = slurm_output_dir / f"manifest_{exp_name}.yaml"
    
    # Extract GPU assignment for this experiment only (convert GpuSlot to dict for YAML)
    exp_gpu_assignment = None
    if gpu_assignment and exp_name in gpu_assignment:
        exp_gpu_assignment = {
            model: {'node': slot.node, 'gpu_index': slot.gpu_index}
            for model, slot in gpu_assignment[exp_name].items()
        }
    
    manifest = {
        'experiment': exp_name,
        'generated_at': datetime.now().isoformat(),
        'gpu_assignment': exp_gpu_assignment,
        'scripts': scripts,
    }
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False)
    
    print(f"Manifest: {manifest_path}")
    return scripts


def parse_gpu_list(gpu_str: str) -> List[int]:
    """Parse GPU list string like '0,1,2' or '0-3' to list of ints."""
    gpus = []
    for part in gpu_str.split(','):
        if '-' in part:
            start, end = part.split('-', 1)
            gpus.extend(range(int(start), int(end) + 1))
        else:
            gpus.append(int(part))
    return gpus


def parse_gpu_spec(gpu_spec: str, default_node: str = "ossc9424vm1") -> Dict[str, List[int]]:
    """
    Parse GPU specification string to node -> GPU list mapping.
    
    Formats supported:
        "0,1,2"                           -> {"ossc9424vm1": [0, 1, 2]}
        "0-3"                             -> {"ossc9424vm1": [0, 1, 2, 3]}
        "ossc9424vm1:0,1,2"               -> {"ossc9424vm1": [0, 1, 2]}
        "ossc9424vm1:0,1;ossc9424vm2:2,3" -> {"ossc9424vm1": [0, 1], "ossc9424vm2": [2, 3]}
    
    Args:
        gpu_spec: GPU specification string
        default_node: Default node name when not specified
    
    Returns:
        Dict mapping node name to list of GPU indices
    """
    result = {}
    
    # Check if it contains node specification (has ':')
    if ':' in gpu_spec:
        # Per-node specification
        for node_spec in gpu_spec.split(';'):
            node_spec = node_spec.strip()
            if ':' in node_spec:
                node_name, gpu_list = node_spec.split(':', 1)
                result[node_name.strip()] = parse_gpu_list(gpu_list.strip())
    else:
        # Simple GPU list - use default node
        result[default_node] = parse_gpu_list(gpu_spec)
    
    return result


def flatten_gpu_spec(gpu_spec: Dict[str, List[int]]) -> List[tuple]:
    """
    Flatten GPU spec to list of (node, gpu_index) tuples.
    
    Example:
        {"ossc9424vm1": [0, 1], "ossc9424vm2": [2, 3]}
        -> [("ossc9424vm1", 0), ("ossc9424vm1", 1), ("ossc9424vm2", 2), ("ossc9424vm2", 3)]
    """
    result = []
    for node, gpus in gpu_spec.items():
        for gpu_idx in gpus:
            result.append((node, gpu_idx))
    return result


@dataclass
class GpuSlot:
    """Represents a GPU slot (node + GPU index)."""
    node: str
    gpu_index: int
    
    def __str__(self):
        return f"{self.node}:GPU{self.gpu_index}"


def create_gpu_assignment(
    models: List[str], 
    experiments: List[Dict], 
    gpu_spec: Dict[str, List[int]]
) -> Dict[str, Dict[str, GpuSlot]]:
    """
    Create model-wise GPU assignment for each experiment.
    
    When submitting an experiment, different models run on different GPUs.
    Jobs on the same GPU run sequentially (via SLURM dependencies).
    
    Example with 5 models, 4 GPU slots across 2 nodes:
        GPU slots: [vm1:0, vm1:1, vm2:0, vm2:1]
        
        model/experiment  e1        e2        e3        e4
        m1                vm1:0     vm1:1     vm2:0     vm2:1
        m2                vm1:1     vm2:0     vm2:1     vm1:0
        m3                vm2:0     vm2:1     vm1:0     vm1:1
        m4                vm2:1     vm1:0     vm1:1     vm2:0
        m5                vm1:0     vm1:1     vm2:0     vm2:1  (cycles back)
    
    When you submit e1: m1->vm1:GPU0, m2->vm1:GPU1, m3->vm2:GPU0, m4->vm2:GPU1, m5->vm1:GPU0 (waits)
    
    Args:
        models: List of model names
        experiments: List of experiment definitions
        gpu_spec: Dict mapping node name to list of GPU indices
    
    Returns:
        Dict[exp_name][model_name] = GpuSlot
    """
    # Flatten GPU spec to list of slots
    gpu_slots = [GpuSlot(node, gpu) for node, gpu in flatten_gpu_spec(gpu_spec)]
    n_slots = len(gpu_slots)
    
    assignment = {}
    
    for exp_idx, exp_def in enumerate(experiments):
        exp_name = exp_def.get('name', f"exp_n{exp_def['n']}_c{exp_def['c']}_h{exp_def['h']}_g{exp_def['g']}")
        assignment[exp_name] = {}
        
        for model_idx, model_name in enumerate(models):
            # Rotate GPU slot assignment based on model index
            # Each experiment shifts the starting slot by exp_idx
            slot_idx = (model_idx + exp_idx) % n_slots
            assignment[exp_name][model_name] = gpu_slots[slot_idx]
    
    return assignment


def generate_from_config(
    config_path: str,
    models_base_dir: Path = None,
    output_base_dir: Path = None,
    slurm_output_dir: Path = None,
    log_dir: str = None,
    gpu_spec: Dict[str, List[int]] = None,
):
    """Generate SLURM scripts from a comprehensive config file.
    
    Args:
        config_path: Path to YAML config file
        models_base_dir: Override for models directory
        output_base_dir: Override for output directory
        slurm_output_dir: Override for SLURM scripts directory
        log_dir: Override for log directory
        gpu_spec: Dict mapping node name to list of GPU indices
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get defaults from config or use standard defaults
    # Note: config.get() returns None if key exists with null value,
    # so we use `or` to fall back to defaults
    script_dir = Path(__file__).parent.parent
    
    if models_base_dir is None:
        cfg_val = config.get('models_base_dir')
        models_base_dir = Path(cfg_val) if cfg_val else (script_dir / "config" / "models")
    
    if output_base_dir is None:
        cfg_val = config.get('output_base_dir')
        output_base_dir = Path(cfg_val) if cfg_val else Path("/projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval")
    
    if slurm_output_dir is None:
        cfg_val = config.get('slurm_output_dir')
        slurm_output_dir = Path(cfg_val) if cfg_val else (script_dir / "slurm_scripts")
    
    if log_dir is None:
        cfg_val = config.get('log_dir')
        log_dir = cfg_val if cfg_val else "/projects/0/prjs1589/stonybrook/logs"
    
    # Print resolved paths for verification
    print(f"Resolved paths:")
    print(f"  models_base_dir: {models_base_dir}")
    print(f"  output_base_dir: {output_base_dir}")
    print(f"  slurm_output_dir: {slurm_output_dir}")
    print(f"  log_dir: {log_dir}")
    print()
    
    models = config.get('models', [])
    experiments = config.get('experiments', [])
    
    if not models:
        print("Error: No models specified in config!")
        return
    if not experiments:
        print("Error: No experiments specified in config!")
        return
    
    print(f"Generating SLURM scripts...")
    print(f"  Models: {len(models)}")
    print(f"  Experiments: {len(experiments)}")
    
    # GPU assignment (model-wise within each experiment)
    gpu_assignment = None
    if gpu_spec:
        gpu_assignment = create_gpu_assignment(models, experiments, gpu_spec)
        print(f"  GPU Assignment (model-wise per experiment):")
        # Print GPU slots
        gpu_slots = flatten_gpu_spec(gpu_spec)
        print(f"  Available GPU slots: {[f'{n}:GPU{g}' for n, g in gpu_slots]}")
        # Print as a table
        print(f"\n  {'Model':<25} " + " ".join(f"e{i:<12}" for i in range(len(experiments))))
        for model_name in models:
            row = f"  {model_name:<25} "
            for exp_def in experiments:
                exp_name = exp_def.get('name', f"exp_n{exp_def['n']}_c{exp_def['c']}_h{exp_def['h']}_g{exp_def['g']}")
                slot = gpu_assignment[exp_name].get(model_name)
                if slot:
                    cell = f"{slot.node[-3:]}:GPU{slot.gpu_index}"
                else:
                    cell = "-"
                row += f"{cell:<13} "
            print(row)
        print()
    else:
        # Check if config has gpu_indices
        gpu_str = config.get('gpu_indices')
        if gpu_str:
            gpu_spec = parse_gpu_spec(str(gpu_str))
            gpu_assignment = create_gpu_assignment(models, experiments, gpu_spec)
            print(f"  GPU Assignment from config (model-wise per experiment):")
            gpu_slots = flatten_gpu_spec(gpu_spec)
            print(f"  Available GPU slots: {[f'{n}:GPU{g}' for n, g in gpu_slots]}")
            # Print as a table
            print(f"\n  {'Model':<25} " + " ".join(f"e{i:<12}" for i in range(len(experiments))))
            for model_name in models:
                row = f"  {model_name:<25} "
                for exp_def in experiments:
                    exp_name = exp_def.get('name', f"exp_n{exp_def['n']}_c{exp_def['c']}_h{exp_def['h']}_g{exp_def['g']}")
                    slot = gpu_assignment[exp_name].get(model_name)
                    if slot:
                        cell = f"{slot.node[-3:]}:GPU{slot.gpu_index}"
                    else:
                        cell = "-"
                    row += f"{cell:<13} "
                print(row)
            print()
    
    print("=" * 60)
    
    all_scripts = []
    
    for exp_def in experiments:
        exp_config = create_experiment_config(
            n=exp_def['n'],
            c=exp_def['c'],
            h=exp_def['h'],
            g=exp_def['g'],
            name=exp_def.get('name'),
            exclude_padding=exp_def.get('exclude_padding', True),
            generation_batch_size=exp_def.get('generation_batch_size', 64),
            compute_by_age=exp_def.get('compute_by_age', False),
            prefix_lengths=exp_def.get('prefix_lengths'),  # Allow explicit override
        )
        
        # Allow experiment-specific overrides
        if 'generation' in exp_def:
            exp_config['generation'].update(exp_def['generation'])
        if 'statistics' in exp_def:
            exp_config['statistics'].update(exp_def['statistics'])
        
        print(f"\nExperiment: {exp_config['experiment_name']}")
        print(f"  n={exp_def['n']}, c={exp_def['c']}, h={exp_def['h']}, g={exp_def['g']}")
        print(f"  prefix_lengths={exp_config['prefix_lengths']}")
        print(f"  exclude_padding={exp_config['exclude_padding']}, batch_size={exp_config['generation_batch_size']}")
        print(f"  compute_by_age={exp_config['compute_by_age']}")
        print("-" * 40)
        
        scripts = generate_slurm_scripts(
            models=models,
            exp_config=exp_config,
            models_base_dir=models_base_dir,
            output_base_dir=output_base_dir,
            slurm_output_dir=slurm_output_dir,
            log_dir=log_dir,
            gpu_assignment=gpu_assignment,
        )
        all_scripts.extend(scripts)
    
    print("=" * 60)
    print(f"Generated {len(all_scripts)} script pairs")
    
    return all_scripts


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for generative evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # With config file (multiple experiments and models)
    python generate_slurm.py --config experiments_config.yaml

    # With GPU assignment on single node (default: ossc9424vm1)
    python generate_slurm.py --config experiments_config.yaml --gpus 0,1,2,3

    # With GPU assignment across multiple nodes
    python generate_slurm.py --config experiments_config.yaml \\
        --gpus "ossc9424vm1:0,1;ossc9424vm2:0,1"

    # With experiment name
    python generate_slurm.py --models model_v1 model_v2 --experiment exp_n10_c100

    # With direct parameters
    python generate_slurm.py --models model_v1 --n 10 --c 100 --h 20 --g 100

GPU Assignment:
    The --gpus flag assigns models to GPU slots round-robin within each experiment.
    Each GPU slot is a (node, gpu_index) pair.
    
    Formats:
      Simple:    '0,1,2' or '0-3'  (default node: ossc9424vm1)
      Per-node:  'ossc9424vm1:0,1,2'
      Multi-node: 'ossc9424vm1:0,1;ossc9424vm2:0,1'
    
    Example with 5 models (m1-m5), 4 GPU slots, and --gpus 0,1,2,3:
    
        model/exp  e1           e2           e3           e4
        m1         vm1:GPU0     vm1:GPU1     vm1:GPU2     vm1:GPU3
        m2         vm1:GPU1     vm1:GPU2     vm1:GPU3     vm1:GPU0
        m3         vm1:GPU2     vm1:GPU3     vm1:GPU0     vm1:GPU1
        m4         vm1:GPU3     vm1:GPU0     vm1:GPU1     vm1:GPU2
        m5         vm1:GPU0     vm1:GPU1     vm1:GPU2     vm1:GPU3  (cycles back)
    
    When you submit e1:
      m1 -> ossc9424vm1:GPU0
      m2 -> ossc9424vm1:GPU1
      m3 -> ossc9424vm1:GPU2
      m4 -> ossc9424vm1:GPU3
      m5 -> ossc9424vm1:GPU0 (waits for m1 via SLURM dependency)

    Generated scripts include:
      - #SBATCH --nodelist=ossc9424vm1  (to specify the node)
      - export CUDA_VISIBLE_DEVICES=0   (to specify the GPU)

Config file format (YAML):
    models:
      - model_v1
      - model_v2
    gpu_indices: "0,1,2"  # Simple format (default node)
    # Or: gpu_indices: "ossc9424vm1:0,1;ossc9424vm2:0,1"  # Multi-node
    experiments:
      - n: 10
        c: 100
        h: 20
        g: 100
      - n: 100
        c: 1000
        h: 20
        g: 100
        """
    )
    
    # Config file option
    parser.add_argument("--config", help="Path to YAML config file with models and experiments")
    
    # Individual experiment options
    parser.add_argument("--models", nargs='+', help="Model names")
    parser.add_argument("--experiment", help="Experiment config name")
    
    # Direct parameters (alternative to experiment config)
    parser.add_argument("--n", type=int, help="Number of people")
    parser.add_argument("--c", type=int, help="Number of generations per person")
    parser.add_argument("--h", type=int, help="Horizon (tokens to generate)")
    parser.add_argument("--g", type=int, help="Prefix gap")
    
    # GPU assignment
    parser.add_argument("--gpus", help="GPU specification. Formats:\n"
                       "  Simple: '0,1,2' or '0-3' (uses default node ossc9424vm1)\n"
                       "  Per-node: 'ossc9424vm1:0,1,2' or 'ossc9424vm1:0,1;ossc9424vm2:2,3'")
    
    # Directory overrides
    parser.add_argument("--models-dir", help="Models configuration directory")
    parser.add_argument("--output-dir", help="Output base directory")
    parser.add_argument("--slurm-dir", help="SLURM scripts output directory")
    parser.add_argument("--log-dir", help="SLURM log directory (default: from config or /projects/0/prjs1589/stonybrook/logs)")
    
    args = parser.parse_args()
    
    # Parse GPU spec if provided
    gpu_spec = None
    if args.gpus:
        gpu_spec = parse_gpu_spec(args.gpus)
        gpu_slots = flatten_gpu_spec(gpu_spec)
        print(f"GPU assignment enabled: {[f'{n}:GPU{g}' for n, g in gpu_slots]}")
    
    # Determine mode
    if args.config:
        # Config file mode
        generate_from_config(
            config_path=args.config,
            models_base_dir=Path(args.models_dir) if args.models_dir else None,
            output_base_dir=Path(args.output_dir) if args.output_dir else None,
            slurm_output_dir=Path(args.slurm_dir) if args.slurm_dir else None,
            log_dir=args.log_dir,
            gpu_spec=gpu_spec,
        )
    elif args.models:
        # Individual experiment mode
        script_dir = Path(__file__).parent.parent
        
        models_base_dir = Path(args.models_dir) if args.models_dir else script_dir / "config" / "models"
        output_base_dir = Path(args.output_dir) if args.output_dir else Path("/projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval")
        slurm_output_dir = Path(args.slurm_dir) if args.slurm_dir else script_dir / "slurm_scripts"
        
        # Get or create experiment config
        if args.experiment:
            exp_base_dir = script_dir / "config" / "experiments"
            exp_config = load_experiment_config(args.experiment, exp_base_dir)
        elif args.n and args.c and args.h and args.g:
            exp_config = create_experiment_config(args.n, args.c, args.h, args.g)
            # Save for reference
            exp_dir = script_dir / "config" / "experiments"
            exp_dir.mkdir(parents=True, exist_ok=True)
            exp_path = exp_dir / f"{exp_config['experiment_name']}.yaml"
            with open(exp_path, 'w') as f:
                yaml.dump(exp_config, f, default_flow_style=False)
            print(f"Created experiment config: {exp_path}")
        else:
            parser.error("Either --experiment or all of --n, --c, --h, --g are required")
        
        print(f"\nExperiment: {exp_config['experiment_name']}")
        print("-" * 40)
        
        # Generate scripts
        generate_slurm_scripts(
            models=args.models,
            exp_config=exp_config,
            models_base_dir=models_base_dir,
            output_base_dir=output_base_dir,
            slurm_output_dir=slurm_output_dir,
            log_dir=args.log_dir,
        )
    else:
        parser.error("Either --config or --models with experiment options are required")


if __name__ == "__main__":
    main()
