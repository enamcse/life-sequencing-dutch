#!/usr/bin/env python3
"""
Setup Model Configuration

Creates model folders with standardized structure for the evaluation pipeline.

Usage:
    # Single model via command line
    python setup_model.py --name model_v1 \
        --checkpoint /path/to/model.ckpt \
        --data /path/to/encoded.h5 \
        --vocab /path/to/vocab.csv

    # Multiple models via config file
    python setup_model.py --config models_config.yaml

Config file format (YAML):
    models:
      - name: model_v1
        checkpoint: /path/to/model1.ckpt
        data: /path/to/encoded.h5
        vocab: /path/to/vocab.csv
      - name: model_v2
        checkpoint: /path/to/model2.ckpt
        data: /path/to/encoded.h5
        vocab: /path/to/vocab.csv
"""

import argparse
import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional


def setup_single_model(
    name: str,
    checkpoint: str,
    data: str,
    vocab: str,
    base_dir: Path,
    output_base: str,
) -> Path:
    """
    Create a model configuration folder.
    
    Args:
        name: Model name (folder name)
        checkpoint: Path to model checkpoint
        data: Path to encoded HDF5 data
        vocab: Path to vocabulary CSV
        base_dir: Base directory for model configs
        output_base: Base directory for outputs
    
    Returns:
        Path to created model directory
    """
    model_dir = base_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symlink to checkpoint
    ckpt_link = model_dir / "model.ckpt"
    if ckpt_link.exists() or ckpt_link.is_symlink():
        ckpt_link.unlink()
    ckpt_link.symlink_to(checkpoint)
    
    # Create config file
    config = {
        'model_name': name,
        'checkpoint_path': checkpoint,
        'data_path': data,
        'vocab_path': vocab,
        'output_dir': os.path.join(output_base, name),
    }
    
    config_path = model_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"  ✓ {name}")
    print(f"      Directory: {model_dir}")
    print(f"      Checkpoint: {ckpt_link} -> {checkpoint}")
    print(f"      Config: {config_path}")
    print(f"      Output: {config['output_dir']}")
    
    return model_dir


def setup_from_config(
    config_path: str,
    base_dir: Path = None,
    output_base: str = None,
) -> List[Path]:
    """
    Setup multiple models from a config file.
    
    Args:
        config_path: Path to YAML config file
        base_dir: Base directory for model configs (overrides config)
        output_base: Base directory for outputs (overrides config)
    
    Returns:
        List of created model directories
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get directories from config or use defaults
    if base_dir is None:
        script_dir = Path(__file__).parent.parent
        base_dir = Path(config.get('models_base_dir', script_dir / "config" / "models"))
    
    if output_base is None:
        output_base = config.get('output_base_dir', "/projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval")
    
    models = config.get('models', [])
    if not models:
        print("No models found in config file!")
        return []
    
    print(f"Setting up {len(models)} models from config...")
    print("-" * 60)
    
    created_dirs = []
    for model in models:
        model_dir = setup_single_model(
            name=model['name'],
            checkpoint=model['checkpoint'],
            data=model['data'],
            vocab=model['vocab'],
            base_dir=base_dir,
            output_base=output_base,
        )
        created_dirs.append(model_dir)
        print()
    
    print("-" * 60)
    print(f"Setup complete! Created {len(created_dirs)} model configurations.")
    
    return created_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Setup model configuration folder(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single model via command line
    python setup_model.py --name model_v1 \\
        --checkpoint /path/to/model.ckpt \\
        --data /path/to/encoded.h5 \\
        --vocab /path/to/vocab.csv

    # Multiple models via config file
    python setup_model.py --config models_config.yaml

Config file format (YAML):
    models:
      - name: model_v1
        checkpoint: /path/to/model1.ckpt
        data: /path/to/encoded.h5
        vocab: /path/to/vocab.csv
      - name: model_v2
        checkpoint: /path/to/model2.ckpt
        data: /path/to/encoded.h5
        vocab: /path/to/vocab.csv
        """
    )
    
    # Config file option
    parser.add_argument("--config", help="Path to YAML config file with model definitions")
    
    # Single model options
    parser.add_argument("--name", help="Model name")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    parser.add_argument("--data", help="Path to encoded HDF5 data")
    parser.add_argument("--vocab", help="Path to vocabulary CSV")
    
    # Directory overrides
    parser.add_argument("--base-dir", help="Base directory for model configs")
    parser.add_argument("--output-base", help="Base directory for outputs")
    
    args = parser.parse_args()
    
    # Determine mode
    if args.config:
        # Config file mode
        base_dir = Path(args.base_dir) if args.base_dir else None
        setup_from_config(args.config, base_dir, args.output_base)
    elif args.name and args.checkpoint and args.data and args.vocab:
        # Single model mode
        if args.base_dir:
            base_dir = Path(args.base_dir)
        else:
            script_dir = Path(__file__).parent.parent
            base_dir = script_dir / "config" / "models"
        
        output_base = args.output_base or "/projects/0/prjs1589/stonybrook/llm/gen_out/gen_eval"
        
        print("Setting up single model...")
        print("-" * 60)
        setup_single_model(
            name=args.name,
            checkpoint=args.checkpoint,
            data=args.data,
            vocab=args.vocab,
            base_dir=base_dir,
            output_base=output_base,
        )
        print()
        print("-" * 60)
        print("Setup complete!")
    else:
        parser.error("Either --config or all of --name, --checkpoint, --data, --vocab are required")


if __name__ == "__main__":
    main()
