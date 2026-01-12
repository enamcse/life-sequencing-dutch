#!/usr/bin/env python3
"""
Mortality Prediction Label Generation

Generates mortality prediction labels based on background and death registry data.

Key Dates:
    - Genesis Date: December 30, 1971
    - Cutoff Date: December 31, 2020
    - Observation Window: Jan 1, 2021 to Dec 31, 2023 (3 years)
    - Max available daysSinceFirstEvent: 18817 (July 7, 2023)

Label Logic:
    For persons alive after Dec 31, 2020:
    - If CUTOFFDATE minus BIRTHDATE produces age 0-100:
        - If death file does not have an entry: Label 0
        - Elif death file has entry with daysSinceFirstEvent in [17534, 18995): Label 1
        - Else: discard
    - Else: discard

daysSinceFirstEvent reference:
    - 17534 = Jan 1, 2021 (inclusive)
    - 18995 = Dec 31, 2023 (exclusive, i.e., up to Dec 30, 2023)

Output:
    - all/death-after-2020.parquet: Full dataset with RINPERSOON, is_dead
    - subset/death-after-2020.parquet: Random 200k sample (or all if less)
    - all-splits/{train,val,test}/death-after-2020.parquet: 70:10:20 split
    - subset-splits/{train,val,test}/death-after-2020.parquet: 70:10:20 split
    - stats/mortality_statistics.csv: Statistics summary
    - stats/plots/: Various visualization plots

Usage:
    python generate_mortality_labels.py \\
        --background-file /path/to/background.parquet \\
        --death-file /path/to/death.parquet \\
        --output-dir /path/to/output \\
        --stats-dir /path/to/stats
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm


# ============================================================================
# CONSTANTS
# ============================================================================

# Genesis date: December 30, 1971
GENESIS_DATE = datetime(1971, 12, 30)

# Cutoff date: December 31, 2020
CUTOFF_DATE = datetime(2020, 12, 31)

# Observation window
OBSERVATION_START = datetime(2021, 1, 1)
OBSERVATION_END = datetime(2023, 12, 31)

# daysSinceFirstEvent boundaries
# Jan 1, 2021 = (2021-01-01) - (1971-12-30) = 17899 days? Let's calculate properly
# Actually, let's compute: 
#   Jan 1, 2021 - Dec 30, 1971 = 49 years + 2 days
#   = 17899 days approximately
# But user specified: 17534 (inclusive) to 18995 (exclusive)
# Let's use the user-provided values
DAYS_OBSERVATION_START = 17534  # Jan 1, 2021 (inclusive)
DAYS_OBSERVATION_END = 18995    # Dec 31, 2023 (exclusive)

# Age constraints at cutoff date
MIN_AGE = 0
MAX_AGE = 100

# Subset sample size
SUBSET_SIZE = 200000

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Random seed for reproducibility
RANDOM_SEED = 42

# Output filename
OUTPUT_FILENAME = "death-after-2020.parquet"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_birthdate(year: int, month: int, day: int = 1) -> datetime:
    """Compute birthdate from year, month, and assumed day=1."""
    return datetime(int(year), int(month), day)


def compute_age_at_cutoff(birthdate: datetime, cutoff: datetime = CUTOFF_DATE) -> float:
    """Compute age in years at the cutoff date."""
    delta = cutoff - birthdate
    return delta.days / 365.25


def days_since_genesis(date: datetime) -> float:
    """Compute days since genesis date."""
    return (date - GENESIS_DATE).days


def validate_output_folder(output_dir: Path) -> bool:
    """
    Validate that output folder ends with 'all'.
    
    Args:
        output_dir: Path to the output directory containing 'all' folder
        
    Returns:
        True if valid, raises error otherwise
    """
    all_folder = output_dir / "all"
    if not all_folder.name == "all":
        raise ValueError(
            f"Output folder structure error: Expected 'all' folder at {all_folder}. "
            f"The 'all' folder should be the last component of the path."
        )
    return True


def load_background_data(filepath: Path) -> pd.DataFrame:
    """
    Load background data file.
    
    Expected columns: RINPERSOON, year, month
    
    Args:
        filepath: Path to background parquet file
        
    Returns:
        DataFrame with columns: RINPERSOON, birth_year, birth_month, birthdate, age_at_cutoff
    """
    print(f"Loading background data from: {filepath}")
    df = pd.read_parquet(filepath)
    
    # Normalize column names (handle case sensitivity)
    df.columns = [col.upper() if col.upper() == 'RINPERSOON' else col.lower() for col in df.columns]
    
    # Ensure required columns exist
    required_cols = ['RINPERSOON', 'year', 'month']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in background file: {col}")
    
    # Compute birthdate (assuming day=1)
    df['birth_year'] = df['year'].astype(int)
    df['birth_month'] = df['month'].astype(int)
    df['birthdate'] = pd.to_datetime(
        df.apply(lambda row: f"{int(row['birth_year'])}-{int(row['birth_month']):02d}-01", axis=1)
    )
    
    # Compute age at cutoff date
    df['age_at_cutoff'] = (CUTOFF_DATE - df['birthdate']).dt.days / 365.25
    
    print(f"  Loaded {len(df):,} records")
    print(f"  Birth years range: {df['birth_year'].min()} - {df['birth_year'].max()}")
    
    return df[['RINPERSOON', 'birth_year', 'birth_month', 'birthdate', 'age_at_cutoff']]


def load_death_data(filepath: Path) -> pd.DataFrame:
    """
    Load death registry data file.
    
    Expected columns: RINPERSOON, daysSinceFirstEvent, age
    
    Args:
        filepath: Path to death parquet file
        
    Returns:
        DataFrame with death information
    """
    print(f"Loading death data from: {filepath}")
    df = pd.read_parquet(filepath)
    
    # Normalize column names
    col_mapping = {}
    for col in df.columns:
        if col.upper() == 'RINPERSOON':
            col_mapping[col] = 'RINPERSOON'
        elif col.lower() in ['dayssincefirstevent', 'days_since_first_event']:
            col_mapping[col] = 'daysSinceFirstEvent'
        elif col.lower() == 'age':
            col_mapping[col] = 'age'
    
    df = df.rename(columns=col_mapping)
    
    # Ensure required columns exist
    required_cols = ['RINPERSOON', 'daysSinceFirstEvent', 'age']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in death file: {col}")
    
    print(f"  Loaded {len(df):,} death records")
    print(f"  daysSinceFirstEvent range: {df['daysSinceFirstEvent'].min():.2f} - {df['daysSinceFirstEvent'].max():.2f}")
    print(f"  Age at death range: {df['age'].min():.2f} - {df['age'].max():.2f}")
    
    return df[['RINPERSOON', 'daysSinceFirstEvent', 'age']]


def generate_labels(
    background_df: pd.DataFrame, 
    death_df: pd.DataFrame
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate mortality labels based on the specified logic.
    
    Logic:
        - If age at cutoff is 0-100:
            - If no death record: Label 0
            - If death in observation window [17534, 18995): Label 1
            - Else: discard (died before 2021 or after observation window)
        - Else: discard (age out of range)
    
    Args:
        background_df: Background data with RINPERSOON and age_at_cutoff
        death_df: Death data with RINPERSOON and daysSinceFirstEvent
        
    Returns:
        Tuple of (labeled DataFrame, statistics dict)
    """
    print("\nGenerating mortality labels...")
    
    stats = {
        'total_background': len(background_df),
        'total_deaths': len(death_df),
    }
    
    # Step 1: Filter by age at cutoff (0-100)
    valid_age_mask = (background_df['age_at_cutoff'] >= MIN_AGE) & (background_df['age_at_cutoff'] <= MAX_AGE)
    df = background_df[valid_age_mask].copy()
    stats['valid_age_count'] = len(df)
    stats['discarded_age_count'] = len(background_df) - len(df)
    print(f"  Valid age (0-100) at cutoff: {len(df):,} ({len(df)/len(background_df)*100:.2f}%)")
    print(f"  Discarded (age out of range): {stats['discarded_age_count']:,}")
    
    # Step 2: Create death lookup (only deaths in observation window)
    deaths_in_window = death_df[
        (death_df['daysSinceFirstEvent'] >= DAYS_OBSERVATION_START) & 
        (death_df['daysSinceFirstEvent'] < DAYS_OBSERVATION_END)
    ].copy()
    stats['deaths_in_window'] = len(deaths_in_window)
    print(f"  Deaths in observation window [{DAYS_OBSERVATION_START}, {DAYS_OBSERVATION_END}): {len(deaths_in_window):,}")
    
    # Create set of people who died in window
    died_in_window = set(deaths_in_window['RINPERSOON'].unique())
    
    # Create set of people who died outside window (before 2021 or after observation)
    deaths_before = death_df[death_df['daysSinceFirstEvent'] < DAYS_OBSERVATION_START]
    deaths_after = death_df[death_df['daysSinceFirstEvent'] >= DAYS_OBSERVATION_END]
    died_before_window = set(deaths_before['RINPERSOON'].unique())
    died_after_window = set(deaths_after['RINPERSOON'].unique())
    
    stats['deaths_before_window'] = len(died_before_window)
    stats['deaths_after_window'] = len(died_after_window)
    print(f"  Deaths before observation window: {len(died_before_window):,}")
    print(f"  Deaths after observation window: {len(died_after_window):,}")
    
    # Step 3: Assign labels
    # First, exclude those who died before window (they were not alive after Dec 31, 2020)
    df = df[~df['RINPERSOON'].isin(died_before_window)]
    stats['alive_after_cutoff'] = len(df)
    print(f"  Persons alive after cutoff date: {len(df):,}")
    
    # Assign labels
    df['is_dead'] = df['RINPERSOON'].isin(died_in_window).astype(int)
    
    # Merge death info for those who died in window
    df = df.merge(
        deaths_in_window[['RINPERSOON', 'daysSinceFirstEvent', 'age']].rename(
            columns={'daysSinceFirstEvent': 'death_days', 'age': 'age_at_death'}
        ),
        on='RINPERSOON',
        how='left'
    )
    
    # Compute death date for statistics
    df['death_date'] = df['death_days'].apply(
        lambda x: GENESIS_DATE + timedelta(days=x) if pd.notna(x) else pd.NaT
    )
    
    stats['final_count'] = len(df)
    stats['label_1_count'] = df['is_dead'].sum()
    stats['label_0_count'] = len(df) - df['is_dead'].sum()
    stats['label_1_ratio'] = stats['label_1_count'] / len(df) if len(df) > 0 else 0
    stats['label_0_ratio'] = stats['label_0_count'] / len(df) if len(df) > 0 else 0
    
    print(f"\nFinal dataset:")
    print(f"  Total: {stats['final_count']:,}")
    print(f"  Label 1 (dead): {stats['label_1_count']:,} ({stats['label_1_ratio']*100:.2f}%)")
    print(f"  Label 0 (alive): {stats['label_0_count']:,} ({stats['label_0_ratio']*100:.2f}%)")
    
    return df, stats


def create_subset(df: pd.DataFrame, size: int = SUBSET_SIZE, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create a random subset of the data."""
    if len(df) <= size:
        print(f"  Dataset size ({len(df):,}) <= subset size ({size:,}), using all data")
        return df.copy()
    
    print(f"  Sampling {size:,} rows from {len(df):,}")
    return df.sample(n=size, random_state=seed).reset_index(drop=True)


def split_data(
    df: pd.DataFrame, 
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        train_ratio: Training set ratio (default 0.70)
        val_ratio: Validation set ratio (default 0.10)
        test_ratio: Test set ratio (default 0.20)
        seed: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    np.random.seed(seed)
    n = len(df)
    indices = np.random.permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    return train_df, val_df, test_df


def save_parquet(df: pd.DataFrame, filepath: Path, columns: list = ['RINPERSOON', 'is_dead']):
    """Save DataFrame to parquet with specified columns."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_parquet(filepath, index=False)
    print(f"  Saved: {filepath} ({len(df):,} rows)")


def compute_detailed_statistics(
    df: pd.DataFrame,
    all_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    all_splits: dict,
    subset_splits: dict,
    initial_stats: dict
) -> pd.DataFrame:
    """
    Compute detailed statistics for all output files.
    
    Args:
        df: Full labeled DataFrame (with all columns)
        all_df: All data output
        subset_df: Subset data output
        all_splits: Dict with train, val, test DataFrames from all
        subset_splits: Dict with train, val, test DataFrames from subset
        initial_stats: Initial statistics from label generation
        
    Returns:
        DataFrame with statistics
    """
    stats_rows = []
    
    datasets = {
        'all': all_df,
        'subset': subset_df,
        'all-splits/train': all_splits['train'],
        'all-splits/val': all_splits['val'],
        'all-splits/test': all_splits['test'],
        'subset-splits/train': subset_splits['train'],
        'subset-splits/val': subset_splits['val'],
        'subset-splits/test': subset_splits['test'],
    }
    
    for name, data in datasets.items():
        total = len(data)
        label_1 = data['is_dead'].sum()
        label_0 = total - label_1
        
        stats_rows.append({
            'dataset': name,
            'total_count': total,
            'label_1_count': label_1,
            'label_0_count': label_0,
            'label_1_ratio': label_1 / total if total > 0 else 0,
            'label_0_ratio': label_0 / total if total > 0 else 0,
        })
    
    stats_df = pd.DataFrame(stats_rows)
    return stats_df


def generate_plots(df: pd.DataFrame, stats_dir: Path):
    """
    Generate visualization plots for mortality data.
    
    Args:
        df: Full labeled DataFrame with death information
        stats_dir: Directory to save plots
    """
    plots_dir = stats_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Filter to only deaths for some plots
    deaths_df = df[df['is_dead'] == 1].copy()
    
    if len(deaths_df) == 0:
        print("  Warning: No deaths in data, skipping death-specific plots")
        return
    
    # 1. Label Distribution Pie Chart
    print("  Generating label distribution plot...")
    fig, ax = plt.subplots(figsize=(8, 8))
    labels = ['Alive (0)', 'Dead (1)']
    sizes = [len(df) - len(deaths_df), len(deaths_df)]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0, 0.05)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%',
           shadow=True, startangle=90)
    ax.set_title('Mortality Label Distribution\n(Jan 2021 - Dec 2023)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "label_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Deaths Over Time (by month)
    print("  Generating deaths over time plot...")
    if 'death_date' in deaths_df.columns and deaths_df['death_date'].notna().any():
        deaths_df['death_year'] = deaths_df['death_date'].dt.year
        deaths_df['death_month'] = deaths_df['death_date'].dt.month
        deaths_df['death_year_month'] = deaths_df['death_date'].dt.to_period('M')
        
        monthly_deaths = deaths_df.groupby('death_year_month').size()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        monthly_deaths.plot(kind='bar', ax=ax, color='#e74c3c', alpha=0.8)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Number of Deaths', fontsize=12)
        ax.set_title('Deaths Over Time (Monthly)\nObservation Window: Jan 2021 - Dec 2023', 
                     fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / "deaths_over_time_monthly.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Deaths by Year
        print("  Generating deaths by year plot...")
        yearly_deaths = deaths_df.groupby('death_year').size()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = yearly_deaths.plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Deaths', fontsize=12)
        ax.set_title('Deaths by Year', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, v in enumerate(yearly_deaths.values):
            ax.text(i, v + max(yearly_deaths) * 0.01, f'{v:,}', ha='center', fontsize=11, fontweight='bold')
        
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(plots_dir / "deaths_by_year.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. Age at Death Distribution
    print("  Generating age at death distribution plot...")
    if 'age_at_death' in deaths_df.columns and deaths_df['age_at_death'].notna().any():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(deaths_df['age_at_death'].dropna(), bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Age at Death (years)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Age at Death', fontsize=14, fontweight='bold')
        
        # Add mean and median lines
        mean_age = deaths_df['age_at_death'].mean()
        median_age = deaths_df['age_at_death'].median()
        ax.axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_age:.1f}')
        ax.axvline(median_age, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_age:.1f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "age_at_death_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # 5. Age at Death by Decade (stratified)
    print("  Generating age groups distribution plot...")
    if 'age_at_death' in deaths_df.columns and deaths_df['age_at_death'].notna().any():
        # Create age groups
        age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150]
        age_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', '100+']
        deaths_df['age_group'] = pd.cut(deaths_df['age_at_death'], bins=age_bins, labels=age_labels, right=False)
        
        age_group_counts = deaths_df['age_group'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = age_group_counts.plot(kind='bar', ax=ax, color='#1abc9c', alpha=0.8, edgecolor='black')
        ax.set_xlabel('Age Group', fontsize=12)
        ax.set_ylabel('Number of Deaths', fontsize=12)
        ax.set_title('Deaths by Age Group (Decades)', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, v in enumerate(age_group_counts.values):
            ax.text(i, v + max(age_group_counts) * 0.01, f'{v:,}', ha='center', fontsize=10)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / "deaths_by_age_group.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # 6. Heatmap: Death Count by Year vs Age Group
    print("  Generating heatmap (year vs age group)...")
    if 'death_year' in deaths_df.columns and 'age_group' in deaths_df.columns:
        heatmap_data = deaths_df.groupby(['death_year', 'age_group']).size().unstack(fill_value=0)
        
        # Ensure all age groups are present
        for label in age_labels:
            if label not in heatmap_data.columns:
                heatmap_data[label] = 0
        heatmap_data = heatmap_data[age_labels]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Use log norm for better visualization if values vary widely
        if heatmap_data.values.max() > 100:
            sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                       norm=LogNorm(vmin=max(1, heatmap_data.values.min()), 
                                   vmax=heatmap_data.values.max()),
                       cbar_kws={'label': 'Death Count (log scale)'})
        else:
            sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                       cbar_kws={'label': 'Death Count'})
        
        ax.set_xlabel('Age Group at Death', fontsize=12)
        ax.set_ylabel('Year of Death', fontsize=12)
        ax.set_title('Heatmap: Deaths by Year and Age Group', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "heatmap_year_vs_age.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # 7. Birth Year Distribution (all persons)
    print("  Generating birth year distribution plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # All persons
    ax1 = axes[0]
    df['birth_year'].hist(bins=50, ax=ax1, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Birth Year', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Birth Year Distribution (All Persons)', fontsize=14, fontweight='bold')
    
    # By label
    ax2 = axes[1]
    df[df['is_dead'] == 0]['birth_year'].hist(bins=50, ax=ax2, alpha=0.6, label='Alive', color='#2ecc71')
    df[df['is_dead'] == 1]['birth_year'].hist(bins=50, ax=ax2, alpha=0.6, label='Dead', color='#e74c3c')
    ax2.set_xlabel('Birth Year', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Birth Year Distribution by Label', fontsize=14, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / "birth_year_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 8. Age at Cutoff Distribution
    print("  Generating age at cutoff distribution plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df[df['is_dead'] == 0]['age_at_cutoff'].hist(bins=50, ax=ax, alpha=0.6, label='Alive', color='#2ecc71')
    df[df['is_dead'] == 1]['age_at_cutoff'].hist(bins=50, ax=ax, alpha=0.6, label='Dead', color='#e74c3c')
    ax.set_xlabel('Age at Cutoff Date (Dec 31, 2020)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Age at Cutoff Date Distribution by Label', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / "age_at_cutoff_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 9. Mortality Rate by Age Group
    print("  Generating mortality rate by age group plot...")
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150]
    age_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', '100+']
    df['age_group_cutoff'] = pd.cut(df['age_at_cutoff'], bins=age_bins, labels=age_labels, right=False)
    
    mortality_by_age = df.groupby('age_group_cutoff').agg(
        total=('is_dead', 'count'),
        deaths=('is_dead', 'sum')
    )
    mortality_by_age['mortality_rate'] = mortality_by_age['deaths'] / mortality_by_age['total'] * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    mortality_by_age['mortality_rate'].plot(kind='bar', ax=ax, color='#e74c3c', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Age Group at Cutoff (Dec 31, 2020)', fontsize=12)
    ax.set_ylabel('Mortality Rate (%)', fontsize=12)
    ax.set_title('3-Year Mortality Rate by Age Group\n(Jan 2021 - Dec 2023)', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(mortality_by_age['mortality_rate'].values):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / "mortality_rate_by_age.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 10. Summary Dashboard
    print("  Generating summary dashboard...")
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Label distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sizes = [len(df) - len(deaths_df), len(deaths_df)]
    colors = ['#2ecc71', '#e74c3c']
    ax1.pie(sizes, labels=['Alive', 'Dead'], colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Label Distribution', fontsize=12, fontweight='bold')
    
    # Deaths by year
    ax2 = fig.add_subplot(gs[0, 1])
    if 'death_year' in deaths_df.columns:
        yearly = deaths_df.groupby('death_year').size()
        yearly.plot(kind='bar', ax=ax2, color='#3498db', alpha=0.8)
        ax2.set_title('Deaths by Year', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Count')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
    
    # Age at death distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if 'age_at_death' in deaths_df.columns and deaths_df['age_at_death'].notna().any():
        deaths_df['age_at_death'].hist(bins=30, ax=ax3, color='#9b59b6', alpha=0.7)
        ax3.set_title('Age at Death', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Age')
        ax3.set_ylabel('Frequency')
    
    # Mortality rate by age
    ax4 = fig.add_subplot(gs[1, 0])
    mortality_by_age['mortality_rate'].plot(kind='bar', ax=ax4, color='#e74c3c', alpha=0.8)
    ax4.set_title('Mortality Rate by Age Group', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Age Group')
    ax4.set_ylabel('Rate (%)')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Birth year by label
    ax5 = fig.add_subplot(gs[1, 1])
    df[df['is_dead'] == 0]['birth_year'].hist(bins=30, ax=ax5, alpha=0.6, label='Alive', color='#2ecc71')
    df[df['is_dead'] == 1]['birth_year'].hist(bins=30, ax=ax5, alpha=0.6, label='Dead', color='#e74c3c')
    ax5.set_title('Birth Year by Label', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Birth Year')
    ax5.legend()
    
    # Statistics text box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_text = f"""
    MORTALITY PREDICTION DATASET SUMMARY
    =====================================
    
    Observation Window: Jan 1, 2021 - Dec 31, 2023
    
    Total Records: {len(df):,}
    
    Label Distribution:
      • Alive (0): {len(df) - len(deaths_df):,} ({(len(df) - len(deaths_df))/len(df)*100:.2f}%)
      • Dead (1):  {len(deaths_df):,} ({len(deaths_df)/len(df)*100:.2f}%)
    
    Age Statistics (at cutoff):
      • Mean: {df['age_at_cutoff'].mean():.1f} years
      • Median: {df['age_at_cutoff'].median():.1f} years
      • Min: {df['age_at_cutoff'].min():.1f} years
      • Max: {df['age_at_cutoff'].max():.1f} years
    """
    
    if 'age_at_death' in deaths_df.columns and deaths_df['age_at_death'].notna().any():
        stats_text += f"""
    Age at Death Statistics:
      • Mean: {deaths_df['age_at_death'].mean():.1f} years
      • Median: {deaths_df['age_at_death'].median():.1f} years
    """
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Mortality Prediction Label Dataset - Summary Dashboard', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(plots_dir / "summary_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Plots saved to: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate mortality prediction labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python generate_mortality_labels.py \\
        --background-file /path/to/step2/background.parquet \\
        --death-file /path/to/step2/death.parquet \\
        --output-dir /path/to/labels \\
        --stats-dir /path/to/stats

The output-dir should contain 'all', 'subset', 'all-splits', 'subset-splits' folders.
These folders will NOT be created - they must already exist.
Only the parquet files and subfolders (train/val/test) will be created.
        """
    )
    
    parser.add_argument("--background-file", required=True,
                        help="Path to background parquet file (columns: RINPERSOON, year, month)")
    parser.add_argument("--death-file", required=True,
                        help="Path to death parquet file (columns: RINPERSOON, daysSinceFirstEvent, age)")
    parser.add_argument("--output-dir", required=True,
                        help="Base output directory (must have 'all' as last folder or parent of standard folders)")
    parser.add_argument("--stats-dir", required=True,
                        help="Directory for statistics and plots")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")
    parser.add_argument("--subset-size", type=int, default=SUBSET_SIZE,
                        help=f"Subset sample size (default: {SUBSET_SIZE})")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Parse paths
    background_file = Path(args.background_file)
    death_file = Path(args.death_file)
    output_dir = Path(args.output_dir)
    stats_dir = Path(args.stats_dir)
    
    # Validate input files exist
    if not background_file.exists():
        print(f"Error: Background file not found: {background_file}")
        sys.exit(1)
    if not death_file.exists():
        print(f"Error: Death file not found: {death_file}")
        sys.exit(1)
    
    # The output_dir should be the parent of 'all', 'subset', etc.
    # Check if standard folders exist (don't create them)
    all_folder = output_dir / "all"
    subset_folder = output_dir / "subset"
    all_splits_folder = output_dir / "all-splits"
    subset_splits_folder = output_dir / "subset-splits"
    
    # Validate that folders exist
    for folder in [all_folder, subset_folder, all_splits_folder, subset_splits_folder]:
        if not folder.exists():
            print(f"Error: Required folder does not exist: {folder}")
            print("Please create the folder structure first. This script will NOT create")
            print("'all', 'subset', 'all-splits', or 'subset-splits' folders.")
            sys.exit(1)
    
    # Validate all folder name
    if all_folder.name != "all":
        print(f"Error: Expected 'all' folder, got: {all_folder.name}")
        sys.exit(1)
    
    # Create stats directory
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MORTALITY PREDICTION LABEL GENERATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Background file: {background_file}")
    print(f"  Death file: {death_file}")
    print(f"  Output directory: {output_dir}")
    print(f"  Statistics directory: {stats_dir}")
    print(f"  Random seed: {args.seed}")
    print(f"  Subset size: {args.subset_size:,}")
    print()
    print(f"Date Parameters:")
    print(f"  Genesis date: {GENESIS_DATE.strftime('%Y-%m-%d')}")
    print(f"  Cutoff date: {CUTOFF_DATE.strftime('%Y-%m-%d')}")
    print(f"  Observation window: {OBSERVATION_START.strftime('%Y-%m-%d')} to {OBSERVATION_END.strftime('%Y-%m-%d')}")
    print(f"  Days range: [{DAYS_OBSERVATION_START}, {DAYS_OBSERVATION_END})")
    print()
    
    # Load data
    background_df = load_background_data(background_file)
    death_df = load_death_data(death_file)
    
    # Generate labels
    labeled_df, initial_stats = generate_labels(background_df, death_df)
    
    # Prepare output DataFrames
    output_columns = ['RINPERSOON', 'is_dead']
    all_df = labeled_df[output_columns].copy()
    
    # Create subset
    print("\nCreating subset...")
    subset_df = create_subset(labeled_df, size=args.subset_size, seed=args.seed)[output_columns]
    
    # Split all data
    print("\nSplitting full dataset (70:10:20)...")
    all_train, all_val, all_test = split_data(all_df, seed=args.seed)
    all_splits = {'train': all_train, 'val': all_val, 'test': all_test}
    print(f"  Train: {len(all_train):,}, Val: {len(all_val):,}, Test: {len(all_test):,}")
    
    # Split subset data
    print("\nSplitting subset dataset (70:10:20)...")
    subset_train, subset_val, subset_test = split_data(subset_df, seed=args.seed)
    subset_splits = {'train': subset_train, 'val': subset_val, 'test': subset_test}
    print(f"  Train: {len(subset_train):,}, Val: {len(subset_val):,}, Test: {len(subset_test):,}")
    
    # Save outputs
    print("\nSaving output files...")
    
    # All data
    save_parquet(all_df, all_folder / OUTPUT_FILENAME)
    
    # Subset data
    save_parquet(subset_df, subset_folder / OUTPUT_FILENAME)
    
    # All splits
    for split_name, split_df in all_splits.items():
        split_folder = all_splits_folder / split_name
        split_folder.mkdir(parents=True, exist_ok=True)
        save_parquet(split_df, split_folder / OUTPUT_FILENAME)
    
    # Subset splits
    for split_name, split_df in subset_splits.items():
        split_folder = subset_splits_folder / split_name
        split_folder.mkdir(parents=True, exist_ok=True)
        save_parquet(split_df, split_folder / OUTPUT_FILENAME)
    
    # Compute and save statistics
    print("\nComputing statistics...")
    stats_df = compute_detailed_statistics(
        labeled_df, all_df, subset_df, all_splits, subset_splits, initial_stats
    )
    
    stats_path = stats_dir / "mortality_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Statistics saved: {stats_path}")
    
    # Print statistics table
    print("\n" + "=" * 80)
    print("STATISTICS SUMMARY")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    print("=" * 80)
    
    # Save detailed initial statistics
    initial_stats_path = stats_dir / "generation_stats.csv"
    pd.DataFrame([initial_stats]).to_csv(initial_stats_path, index=False)
    print(f"  Generation stats saved: {initial_stats_path}")
    
    # Generate plots
    if not args.skip_plots:
        print("\nGenerating plots...")
        try:
            generate_plots(labeled_df, stats_dir)
        except Exception as e:
            print(f"  Warning: Plot generation failed: {e}")
            print("  Continuing without plots...")
    
    print("\n" + "=" * 60)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  {all_folder / OUTPUT_FILENAME}")
    print(f"  {subset_folder / OUTPUT_FILENAME}")
    print(f"  {all_splits_folder / 'train' / OUTPUT_FILENAME}")
    print(f"  {all_splits_folder / 'val' / OUTPUT_FILENAME}")
    print(f"  {all_splits_folder / 'test' / OUTPUT_FILENAME}")
    print(f"  {subset_splits_folder / 'train' / OUTPUT_FILENAME}")
    print(f"  {subset_splits_folder / 'val' / OUTPUT_FILENAME}")
    print(f"  {subset_splits_folder / 'test' / OUTPUT_FILENAME}")
    print(f"\nStatistics: {stats_dir}")


if __name__ == "__main__":
    main()
