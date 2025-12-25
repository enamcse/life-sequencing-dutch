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

echo "=========================================="
echo "Generation Job: {model_name}"
echo "Experiment: {exp_name}"
echo "=========================================="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

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
echo "Statistics Job: {model_name}"
echo "Experiment: {exp_name}"
echo "=========================================="
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

# Load environment
cd ~/life-sequencing-dutch/
source requirements/load_venv.sh

# Run statistics computation
python -m pop2vec.llm.src.new_code.gen_eval.src.compute_statistics \\
    --config {config_path}

echo ""
echo "Completed: $(date)"
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
                             exclude_padding: bool = True) -> dict:
    """Create experiment configuration from parameters."""
    if name is None:
        name = f"exp_n{n}_c{c}_h{h}_g{g}"
    
    # Generate prefix lengths: 1, 1+g, 1+2g, ...
    prefix_lengths = [1]
    current = 1 + g
    while current <= 1001:
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
    run_config = {
        **exp_config,
        'model_name': model_config['model_name'],
        'checkpoint_path': model_config['checkpoint_path'],
        'data_path': model_config['data_path'],
        'vocab_path': model_config['vocab_path'],
        'output_dir': str(output_dir),
        'sequences_path': str(output_dir / 'generated_sequences.parquet'),
        'original_sequences_path': str(output_dir / 'original_sequences.parquet'),
        'statistics_path': str(output_dir / 'statistics_full.csv'),
        'statistics_summary_path': str(output_dir / 'statistics_summary.csv'),
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
):
    """Generate SLURM scripts for all models."""
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
        
        # Generate SLURM script for generation (GPU)
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
        })
        
        print(f"  ✓ {model_name}")
        print(f"      Config: {config_path}")
        print(f"      Gen script: {gen_script_path}")
        print(f"      Stats script: {stats_script_path}")
        print()
    
    # Write manifest
    manifest_path = slurm_output_dir / f"manifest_{exp_name}.yaml"
    manifest = {
        'experiment': exp_name,
        'generated_at': datetime.now().isoformat(),
        'scripts': scripts,
    }
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False)
    
    print(f"Manifest: {manifest_path}")
    return scripts


def generate_from_config(
    config_path: str,
    models_base_dir: Path = None,
    output_base_dir: Path = None,
    slurm_output_dir: Path = None,
    log_dir: str = None,
):
    """Generate SLURM scripts from a comprehensive config file."""
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
        )
        
        # Allow experiment-specific overrides
        if 'generation' in exp_def:
            exp_config['generation'].update(exp_def['generation'])
        if 'statistics' in exp_def:
            exp_config['statistics'].update(exp_def['statistics'])
        
        print(f"\nExperiment: {exp_config['experiment_name']}")
        print(f"  n={exp_def['n']}, c={exp_def['c']}, h={exp_def['h']}, g={exp_def['g']}")
        print(f"  exclude_padding={exp_config['exclude_padding']}")
        print("-" * 40)
        
        scripts = generate_slurm_scripts(
            models=models,
            exp_config=exp_config,
            models_base_dir=models_base_dir,
            output_base_dir=output_base_dir,
            slurm_output_dir=slurm_output_dir,
            log_dir=log_dir,
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

    # With experiment name
    python generate_slurm.py --models model_v1 model_v2 --experiment exp_n10_c100

    # With direct parameters
    python generate_slurm.py --models model_v1 --n 10 --c 100 --h 20 --g 100

Config file format (YAML):
    models:
      - model_v1
      - model_v2
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
    
    # Directory overrides
    parser.add_argument("--models-dir", help="Models configuration directory")
    parser.add_argument("--output-dir", help="Output base directory")
    parser.add_argument("--slurm-dir", help="SLURM scripts output directory")
    parser.add_argument("--log-dir", help="SLURM log directory (default: from config or /projects/0/prjs1589/stonybrook/logs)")
    
    args = parser.parse_args()
    
    # Determine mode
    if args.config:
        # Config file mode
        generate_from_config(
            config_path=args.config,
            models_base_dir=Path(args.models_dir) if args.models_dir else None,
            output_base_dir=Path(args.output_dir) if args.output_dir else None,
            slurm_output_dir=Path(args.slurm_dir) if args.slurm_dir else None,
            log_dir=args.log_dir,
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
