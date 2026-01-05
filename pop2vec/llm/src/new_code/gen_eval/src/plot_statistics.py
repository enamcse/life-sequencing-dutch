#!/usr/bin/env python3
"""
Plot Statistics - Visualization Stage for Generative Evaluation Pipeline

Generates sanity check plots comparing real vs simulated token frequencies by decade.
Supports configurable life event tokens (e.g., death, retirement, school).

Usage:
    python plot_statistics.py --config run_config.yaml
    python plot_statistics.py --token_counts token_counts_by_decade.csv --events_config events.yaml

Event Configuration (events.yaml):
    life_events:
      death:
        tokens: [1234, 1235, 1236]  # Token IDs for death-related events
        color: "#e41a1c"
        label: "Death"
      retirement:
        tokens: [2001, 2002, 2003]
        color: "#377eb8"
        label: "Retirement"
      school:
        tokens: [3001, 3002]
        color: "#4daf4a"
        label: "School/Education"

Output:
    - Individual plots per life event (token_freq_<event>_by_decade.png)
    - Combined comparison plot (token_freq_all_events_by_decade.png)
    - Log scale versions (_log suffix)
"""

import argparse
import logging
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Default color palette for life events
DEFAULT_COLORS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#ffff33",  # yellow
    "#a65628",  # brown
    "#f781bf",  # pink
]

# Decade order for proper sorting
DECADE_ORDER = [
    "0-9", "10-19", "20-29", "30-39", "40-49",
    "50-59", "60-69", "70-79", "80-89", "90-99", "100+"
]


def load_events_config(config_path: str) -> Dict:
    """
    Load life events configuration from YAML file.
    
    Expected format:
        life_events:
          event_name:
            tokens: [token_id1, token_id2, ...]
            color: "#hexcolor" (optional)
            label: "Display Label" (optional)
    """
    if not os.path.exists(config_path):
        logger.warning(f"Events config not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('life_events', {})


def load_token_counts(csv_path: str) -> pd.DataFrame:
    """Load token counts spreadsheet."""
    logger.info(f"Loading token counts: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows")
    return df


def aggregate_event_counts(
    df: pd.DataFrame,
    event_tokens: List[int]
) -> pd.DataFrame:
    """
    Aggregate token counts for a life event (sum of all tokens in the event).
    
    Args:
        df: Token counts DataFrame with columns: decade, token_id, simulated_count, real_count
        event_tokens: List of token IDs for this life event
    
    Returns:
        DataFrame with columns: decade, simulated_count, real_count, N_d
    """
    # Filter to only include tokens for this event
    event_df = df[df['token_id'].isin(event_tokens)]
    
    if event_df.empty:
        logger.warning(f"No data found for tokens: {event_tokens}")
        return pd.DataFrame()
    
    # Aggregate by decade
    agg_df = event_df.groupby('decade').agg({
        'simulated_count': 'sum',
        'real_count': 'sum',
        'N_d': 'first'  # N_d is the same for all tokens in a decade
    }).reset_index()
    
    # Sort by decade order
    decade_order_map = {d: i for i, d in enumerate(DECADE_ORDER)}
    agg_df['_order'] = agg_df['decade'].map(decade_order_map)
    agg_df = agg_df.sort_values('_order').drop(columns=['_order'])
    
    return agg_df


def compute_frequency_per_person(df: pd.DataFrame, n_generations: int = 100) -> pd.DataFrame:
    """
    Compute frequency per person for real and simulated.
    
    Real: count / N_d
    Simulated: count / (N_d * n_generations)
    
    This normalizes to "expected events per person-decade".
    """
    result = df.copy()
    result['real_freq'] = result['real_count'] / result['N_d'].replace(0, np.nan)
    result['simulated_freq'] = result['simulated_count'] / (result['N_d'] * n_generations).replace(0, np.nan)
    return result


def plot_event_by_decade(
    df: pd.DataFrame,
    event_name: str,
    event_label: str,
    color: str,
    output_dir: str,
    n_generations: int = 100,
    log_scale: bool = False
):
    """
    Plot real vs simulated frequency for a single life event by decade.
    
    Creates a line plot with two lines:
    - Real frequency (solid line with circles)
    - Simulated frequency (dashed line with triangles)
    """
    if df.empty:
        logger.warning(f"No data for event: {event_name}")
        return
    
    # Compute per-person frequencies
    plot_df = compute_frequency_per_person(df, n_generations)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    decades = plot_df['decade'].tolist()
    real_freq = plot_df['real_freq'].tolist()
    simulated_freq = plot_df['simulated_freq'].tolist()
    
    x = np.arange(len(decades))
    
    # Plot real and simulated
    ax.plot(x, real_freq, 'o-', color=color, linewidth=2, markersize=8, 
            label='Real', alpha=0.9)
    ax.plot(x, simulated_freq, 's--', color=color, linewidth=2, markersize=8,
            label='Simulated', alpha=0.6)
    
    ax.set_xlabel('Age Decade', fontsize=12)
    ax.set_ylabel('Frequency per Person', fontsize=12)
    ax.set_title(f'{event_label}: Real vs Simulated by Age Decade', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(decades, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    
    plt.tight_layout()
    
    suffix = '_log' if log_scale else ''
    output_path = os.path.join(output_dir, f'token_freq_{event_name}_by_decade{suffix}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plot: {output_path}")
    return output_path


def plot_all_events_comparison(
    events_data: Dict[str, Tuple[pd.DataFrame, str, str]],
    output_dir: str,
    n_generations: int = 100,
    log_scale: bool = False
):
    """
    Plot comparison of all life events in a single figure.
    
    Shows real frequencies for all events (solid lines) and simulated (dashed).
    
    Args:
        events_data: Dict of event_name -> (DataFrame, label, color)
    """
    if not events_data:
        logger.warning("No events data to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Real frequencies
    ax_real = axes[0]
    ax_sim = axes[1]
    
    for event_name, (df, label, color) in events_data.items():
        if df.empty:
            continue
        
        plot_df = compute_frequency_per_person(df, n_generations)
        decades = plot_df['decade'].tolist()
        x = np.arange(len(decades))
        
        ax_real.plot(x, plot_df['real_freq'].tolist(), 'o-', color=color, 
                     linewidth=2, markersize=6, label=label, alpha=0.9)
        ax_sim.plot(x, plot_df['simulated_freq'].tolist(), 's-', color=color,
                    linewidth=2, markersize=6, label=label, alpha=0.9)
    
    # Common setup for both axes
    for ax, title in [(ax_real, 'Real Life Events'), (ax_sim, 'Simulated Life Events')]:
        ax.set_xlabel('Age Decade', fontsize=12)
        ax.set_ylabel('Frequency per Person', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(decades, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    
    plt.tight_layout()
    
    suffix = '_log' if log_scale else ''
    output_path = os.path.join(output_dir, f'token_freq_all_events_by_decade{suffix}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison plot: {output_path}")
    return output_path


def plot_real_vs_simulated_scatter(
    events_data: Dict[str, Tuple[pd.DataFrame, str, str]],
    output_dir: str,
    n_generations: int = 100
):
    """
    Scatter plot of real vs simulated frequencies across all decades and events.
    
    Useful for checking overall calibration: points should lie near the diagonal.
    """
    if not events_data:
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    all_real = []
    all_sim = []
    colors = []
    labels = []
    
    for event_name, (df, label, color) in events_data.items():
        if df.empty:
            continue
        
        plot_df = compute_frequency_per_person(df, n_generations)
        
        for _, row in plot_df.iterrows():
            if pd.notna(row['real_freq']) and pd.notna(row['simulated_freq']):
                all_real.append(row['real_freq'])
                all_sim.append(row['simulated_freq'])
                colors.append(color)
                labels.append(f"{label} ({row['decade']})")
    
    if not all_real:
        logger.warning("No data for scatter plot")
        return
    
    # Plot scatter
    for i, (r, s, c, l) in enumerate(zip(all_real, all_sim, colors, labels)):
        ax.scatter(r, s, c=c, s=50, alpha=0.7)
    
    # Diagonal line (perfect calibration)
    max_val = max(max(all_real), max(all_sim)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Calibration')
    
    ax.set_xlabel('Real Frequency', fontsize=12)
    ax.set_ylabel('Simulated Frequency', fontsize=12)
    ax.set_title('Real vs Simulated Frequencies by Decade', fontsize=14)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Legend for event types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
        for label, color in set((l.split(' (')[0], c) for l, c in zip(labels, colors))
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'real_vs_simulated_scatter.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved scatter plot: {output_path}")
    return output_path


def create_default_events_config(vocab_path: str, output_path: str):
    """
    Create a default events config by scanning the vocabulary for common life events.
    
    This is a helper function - the actual token IDs need to be verified manually.
    """
    logger.info(f"Creating default events config from vocabulary: {vocab_path}")
    
    vocab_df = pd.read_csv(vocab_path)
    
    # Common keywords to search for
    event_keywords = {
        'death': ['death', 'died', 'deceased', 'mortality', 'dood'],
        'retirement': ['retirement', 'pension', 'pensioen'],
        'school': ['school', 'education', 'student', 'onderwijs'],
        'marriage': ['marriage', 'married', 'wedding', 'huwelijk'],
        'divorce': ['divorce', 'divorced', 'scheiding'],
        'birth': ['birth', 'born', 'geboorte'],
        'employment': ['employed', 'job', 'work', 'baan', 'werk'],
        'hospital': ['hospital', 'hospitalized', 'ziekenhuis'],
    }
    
    life_events = {}
    color_idx = 0
    
    for event_name, keywords in event_keywords.items():
        matching_tokens = []
        
        for _, row in vocab_df.iterrows():
            token_str = str(row['TOKEN']).lower()
            if any(kw in token_str for kw in keywords):
                matching_tokens.append(int(row['ID']))
        
        if matching_tokens:
            life_events[event_name] = {
                'tokens': matching_tokens[:10],  # Limit to first 10 matches
                'color': DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)],
                'label': event_name.replace('_', ' ').title()
            }
            color_idx += 1
            logger.info(f"  {event_name}: found {len(matching_tokens)} matching tokens")
    
    config = {'life_events': life_events}
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved default events config: {output_path}")
    logger.info("Please review and adjust token IDs manually!")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot statistics - visualization stage for generative evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using run config (will find token_counts CSV automatically)
    python plot_statistics.py --config run_config.yaml
    
    # Specify token counts CSV directly
    python plot_statistics.py --token_counts token_counts_by_decade.csv --events_config events.yaml
    
    # Create default events config from vocabulary
    python plot_statistics.py --create_events_config --vocab vocab.csv --output events.yaml
        """
    )
    
    parser.add_argument("--config", help="Path to run config YAML")
    parser.add_argument("--token_counts", help="Path to token counts CSV")
    parser.add_argument("--events_config", help="Path to life events config YAML")
    parser.add_argument("--output_dir", help="Output directory for plots")
    parser.add_argument("--n_generations", type=int, default=100,
                        help="Number of generations per person (default: 100)")
    parser.add_argument("--create_events_config", action="store_true",
                        help="Create a default events config from vocabulary")
    parser.add_argument("--vocab", help="Path to vocabulary CSV (for --create_events_config)")
    parser.add_argument("--output", help="Output path for events config (for --create_events_config)")
    
    args = parser.parse_args()
    
    # Handle create_events_config mode
    if args.create_events_config:
        if not args.vocab:
            parser.error("--vocab is required with --create_events_config")
        output_path = args.output or 'events_config.yaml'
        create_default_events_config(args.vocab, output_path)
        return
    
    # Determine paths
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = config.get('output_dir', '.')
        n_people = config.get('n_people')
        n_generations = config.get('n_generations', args.n_generations)
        
        # Try to find token counts file
        if args.token_counts:
            token_counts_path = args.token_counts
        else:
            # Look for token_counts_by_decade*.csv in output_dir
            import glob
            pattern = os.path.join(output_dir, 'token_counts_by_decade*.csv')
            matches = glob.glob(pattern)
            if matches:
                token_counts_path = matches[0]
            else:
                logger.error(f"No token counts file found matching: {pattern}")
                return
        
        # Try to find events config
        if args.events_config:
            events_config_path = args.events_config
        else:
            # Look in config directory or output directory
            possible_paths = [
                os.path.join(output_dir, 'events_config.yaml'),
                os.path.join(os.path.dirname(args.config), 'events_config.yaml'),
                'events_config.yaml'
            ]
            events_config_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    events_config_path = path
                    break
    else:
        if not args.token_counts:
            parser.error("Either --config or --token_counts is required")
        
        token_counts_path = args.token_counts
        events_config_path = args.events_config
        output_dir = args.output_dir or os.path.dirname(token_counts_path) or '.'
        n_generations = args.n_generations
    
    # Load token counts
    token_counts_df = load_token_counts(token_counts_path)
    
    # Determine output directory for plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load events config
    if events_config_path:
        life_events = load_events_config(events_config_path)
        logger.info(f"Loaded {len(life_events)} life events from config")
    else:
        logger.warning("No events config found - using empty config")
        logger.info("Use --create_events_config to generate a template")
        life_events = {}
    
    if not life_events:
        logger.warning("No life events configured. Creating example plots with top tokens.")
        
        # Create plots for top 5 most common tokens
        top_tokens = token_counts_df.groupby('token_id').agg({
            'simulated_count': 'sum',
            'real_count': 'sum',
            'token': 'first'
        }).nlargest(5, 'real_count')
        
        color_idx = 0
        for token_id, row in top_tokens.iterrows():
            event_name = f"token_{token_id}"
            life_events[event_name] = {
                'tokens': [int(token_id)],
                'color': DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)],
                'label': str(row['token'])[:30]  # Truncate long tokens
            }
            color_idx += 1
    
    # Process each life event
    events_data = {}
    
    for event_name, event_config in life_events.items():
        tokens = event_config.get('tokens', [])
        color = event_config.get('color', DEFAULT_COLORS[0])
        label = event_config.get('label', event_name.replace('_', ' ').title())
        
        logger.info(f"Processing event: {event_name} ({len(tokens)} tokens)")
        
        # Aggregate counts for this event
        agg_df = aggregate_event_counts(token_counts_df, tokens)
        
        if agg_df.empty:
            continue
        
        events_data[event_name] = (agg_df, label, color)
        
        # Plot individual event (linear and log scale)
        plot_event_by_decade(agg_df, event_name, label, color, plots_dir, 
                            n_generations, log_scale=False)
        plot_event_by_decade(agg_df, event_name, label, color, plots_dir,
                            n_generations, log_scale=True)
    
    # Plot comparison of all events
    if events_data:
        plot_all_events_comparison(events_data, plots_dir, n_generations, log_scale=False)
        plot_all_events_comparison(events_data, plots_dir, n_generations, log_scale=True)
        plot_real_vs_simulated_scatter(events_data, plots_dir, n_generations)
    
    logger.info("="*60)
    logger.info("Plotting Complete!")
    logger.info(f"  Output directory: {plots_dir}")
    logger.info(f"  Events plotted: {len(events_data)}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
