#!/usr/bin/env python3
"""
Export for Plotting v2

Outputs per dataset:
1. comparisons_{dataset}.csv - 12 comparison row types with num/den per experiment
2. blockwise_{dataset}.csv - Token frequencies by prefix_len (no row_type, no den)
3. decade_{dataset}.csv - Token frequencies by decade (no row_type, no den)

Global output:
4. decade_summary.csv - Merged decade_summary from all experiments with dataset column

Usage: python export_for_plotting_v2.py --output-dir /path/to/gen_eval --export-dir /path/to/export
"""
import argparse, os, re
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import pyarrow.parquet as pq

def parse_exp(name):
    """Extract (short_name, dataset) from experiment folder name."""
    m = re.match(r'(exp_n\d+_c\d+_h\d+_g\d+_k\w+_t\d+_(\w+))_(GD\w*)$', name)
    if not m:
        return (name, None)
    short_name, model_part, dataset = m.group(1), m.group(2), m.group(3)
    if 'bd' in model_part.lower() and not dataset.startswith('GDB'):
        dataset = 'GDB' + dataset[2:]
    return (short_name, dataset)

def load_real_counts(orig_path, ages_path, h=20):
    """Compute real token counts from original_sequences.
    
    For each prefix_len, counts tokens in positions [prefix_len, prefix_len+h).
    Returns (plen_counts, dec_counts) dicts.
    """
    try:
        orig = pq.read_table(orig_path).to_pandas()
        ages = pq.read_table(ages_path).to_pandas()
        print(f"    Loading real counts from {len(orig)} sequences...")
    except Exception as e:
        print(f"    Error loading parquet: {e}")
        return {}, {}
    
    try:
        orig = orig[~orig['is_buddy']]
        ages = ages[~ages['is_buddy']]
        
        plen_counts = defaultdict(lambda: defaultdict(int))
        dec_counts = defaultdict(lambda: defaultdict(int))
        plens = [7, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        
        n_processed = 0
        for _, r in orig.iterrows():
            toks = list(map(int, str(r['original_sequence']).split(',')))
            rlen = int(r['real_length'])
            ar = ages[ages['local_idx'] == r['local_idx']]
            if len(ar) == 0: continue
            ag = list(map(int, str(ar.iloc[0]['age_stream']).split(',')))
            
            for p in plens:
                if p + h > rlen: continue
                for i in range(p, min(p + h, rlen)):
                    tid = toks[i]
                    plen_counts[p][tid] += 1
            
            # For decade counts, use all tokens (not just by prefix_len)
            for i, tid in enumerate(toks[:rlen]):
                age = ag[i] if i < len(ag) else (ag[-1] if ag else 0)
                dec_counts[f"{(age//10)*10}s"][tid] += 1
            
            n_processed += 1
            if n_processed % 1000 == 0:
                print(f"      Processed {n_processed} sequences...")
        
        print(f"    Computed real counts: {len(plen_counts)} prefix_lens, {len(dec_counts)} decades")
        return dict(plen_counts), dict(dec_counts)
    except Exception as e:
        import traceback
        print(f"    Error processing sequences: {e}")
        traceback.print_exc()
        return {}, {}

def load_real_from_token_counts(d):
    """Load real counts from token_counts_by_decade_n*_c*.csv file."""
    for f in d.glob('token_counts_by_decade_n*_c*.csv'):
        df = pd.read_csv(f)
        if 'decade' not in df.columns or 'token_id' not in df.columns:
            continue
        real_col = [c for c in df.columns if 'real' in c.lower()]
        if not real_col: continue
        result = defaultdict(dict)
        for _, row in df.iterrows():
            result[row['decade']][int(row['token_id'])] = int(row[real_col[0]])
        return dict(result)
    return None

def load_comparisons(path, exp_name):
    """Load comparison rows (12 row types) from stats CSV with experiment name."""
    df = pd.read_csv(path)
    comp = df[df['row_type'] != 'token_frequency'][['prefix_len', 'row_type', 'total_num', 'total_den']].copy()
    comp['experiment'] = exp_name
    return comp.rename(columns={'total_num': 'numerator', 'total_den': 'denominator'})

def load_token_freq_block(path, suffix):
    """Load token_frequency rows from stats CSV (numerator only)."""
    df = pd.read_csv(path)
    
    # Debug: print unique row_types
    row_types = df['row_type'].unique().tolist()
    
    tf = df[df['row_type'] == 'token_frequency'][['prefix_len', 'token_id', 'total_num']].copy()
    
    if len(tf) == 0:
        print(f"    WARNING: No token_frequency rows in {path.name}")
        print(f"    Available row_types: {row_types}")
        return pd.DataFrame(columns=['prefix_len', 'token_id', f'num_{suffix}'])
    
    tf = tf.dropna(subset=['token_id'])
    tf['token_id'] = tf['token_id'].astype(int)
    
    # Debug: print stats
    print(f"    {suffix}: {len(tf)} token_freq rows, non-zero: {(tf['total_num'] > 0).sum()}")
    
    return tf.rename(columns={'total_num': f'num_{suffix}'})

def load_block_summary(path, exp_name):
    """Load denominator per prefix_len from stats CSV (one row per prefix_len)."""
    df = pd.read_csv(path)
    # Get one row per prefix_len from token_frequency rows (all have same denominator)
    tf = df[df['row_type'] == 'token_frequency'][['prefix_len', 'total_den']].copy()
    tf = tf.drop_duplicates(subset=['prefix_len'])
    tf['experiment'] = exp_name
    return tf.rename(columns={'total_den': 'denominator'})

def load_token_freq_decade(path, suffix):
    """Load by-age stats CSV (numerator only)."""
    df = pd.read_csv(path)
    cols = ['decade', 'token_id', 'total_num']
    df = df[[c for c in cols if c in df.columns]].copy()
    df = df.dropna(subset=['token_id'])
    df['token_id'] = df['token_id'].astype(int)
    return df.rename(columns={'total_num': f'num_{suffix}'})

def load_decade_summary(path, dataset, exp_name):
    """Load decade_summary CSV and add dataset/experiment columns."""
    df = pd.read_csv(path)
    df['dataset'] = dataset
    df['experiment'] = exp_name
    # Drop expected_* columns (duplicates of total_*)
    drop_cols = [c for c in df.columns if c.startswith('expected_')]
    df = df.drop(columns=drop_cols, errors='ignore')
    return df

def merge_all(dfs, key_cols):
    """Merge list of dataframes on key columns."""
    if not dfs: 
        print(f"    merge_all: No dataframes to merge!")
        return pd.DataFrame()
    
    print(f"    merge_all: {len(dfs)} dataframes, key_cols={key_cols}")
    for i, df in enumerate(dfs):
        print(f"      df[{i}]: {len(df)} rows, cols={list(df.columns)}")
    
    m = dfs[0]
    for i, df in enumerate(dfs[1:], 1):
        before = len(m)
        m = m.merge(df, on=key_cols, how='outer')
        print(f"      After merge {i}: {before} -> {len(m)} rows")
    
    print(f"    merge_all result: {len(m)} rows")
    return m

def add_real_block(merged, real_counts):
    """Add real_count column for blockwise."""
    if not real_counts: 
        print("    add_real_block: No real counts available")
        return merged
    rows = []
    for plen, tcounts in real_counts.items():
        for tid, cnt in tcounts.items():
            rows.append({'prefix_len': plen, 'token_id': tid, 'real_count': cnt})
    if not rows: 
        print("    add_real_block: Empty rows")
        return merged
    rdf = pd.DataFrame(rows)
    print(f"    add_real_block: Adding {len(rdf)} real count entries")
    result = merged.merge(rdf, on=['prefix_len', 'token_id'], how='left')
    n_with_real = result['real_count'].notna().sum()
    print(f"    add_real_block: {n_with_real} / {len(result)} rows have real counts")
    return result

def add_real_decade(merged, real_counts):
    """Add real_count column for decade stats."""
    if not real_counts: 
        print("    add_real_decade: No real counts available")
        return merged
    rows = []
    for dec, tcounts in real_counts.items():
        for tid, cnt in tcounts.items():
            rows.append({'decade': dec, 'token_id': tid, 'real_count': cnt})
    if not rows: 
        print("    add_real_decade: Empty rows")
        return merged
    rdf = pd.DataFrame(rows)
    print(f"    add_real_decade: Adding {len(rdf)} real count entries")
    result = merged.merge(rdf, on=['decade', 'token_id'], how='outer')
    n_with_real = result['real_count'].notna().sum()
    print(f"    add_real_decade: {n_with_real} / {len(result)} rows have real counts")
    return result

def clean_comparisons(df, path):
    """Save comparisons CSV (long format: one row per experiment/prefix/row_type)."""
    # Reorder columns
    df = df[['experiment', 'prefix_len', 'row_type', 'numerator', 'denominator']].copy()
    df['numerator'] = df['numerator'].fillna(0).astype(int)
    df['denominator'] = df['denominator'].fillna(0).astype(int)
    df = df.sort_values(['experiment', 'prefix_len', 'row_type'])
    df.to_csv(path, index=False)
    return len(df)

def clean_token_freq(df, path, key_cols):
    """
    Save token frequency CSV.
    - Replace 0 with empty string in num_* and real_count columns
    - Drop rows where ALL num_* are empty/0
    """
    num_cols = [c for c in df.columns if c.startswith('num_')]
    real_col = 'real_count' if 'real_count' in df.columns else None
    
    print(f"    clean_token_freq: {len(df)} rows, {len(num_cols)} num columns, real_col={real_col is not None}")
    
    if len(df) == 0:
        print(f"    WARNING: Empty dataframe, saving header only")
        df.to_csv(path, index=False)
        return 0
    
    # Fill NaN with 0 first
    df[num_cols] = df[num_cols].fillna(0)
    if real_col:
        df[real_col] = df[real_col].fillna(0)
    
    # Drop rows where ALL num_* are 0
    all_zero = (df[num_cols] == 0).all(axis=1)
    n_zero = all_zero.sum()
    print(f"    Rows with all zeros: {n_zero} / {len(df)}")
    
    df = df[~all_zero].copy()
    
    if len(df) == 0:
        print(f"    WARNING: All rows filtered out!")
        # Save with at least header
        df.to_csv(path, index=False)
        return 0
    
    # Convert to int, then replace 0 with empty string for num columns AND real_count
    for c in num_cols:
        df[c] = df[c].astype(int).astype(str).replace('0', '')
    if real_col:
        df[real_col] = df[real_col].astype(int).astype(str).replace('0', '')
    
    df = df.sort_values(key_cols)
    df.to_csv(path, index=False)
    return len(df)

def export_dataset(ds, dirs, export_dir, decade_summaries, block_summaries):
    """Export files for one dataset."""
    print(f"\n=== {ds}: {len(dirs)} experiments ===")
    
    comp_dfs, block_dfs, dec_dfs = [], [], []
    real_plen, real_dec = None, None
    
    for d in tqdm(dirs, desc=ds):
        short, _ = parse_exp(d.name)
        
        # Find stats files
        stats_file = None
        for f in d.glob('statistics_n*_c*_summary.csv'):
            if 'by_age' not in f.name:
                stats_file = f; break
        
        by_age_file = None
        for f in d.glob('statistics_by_age_n*_c*_summary.csv'):
            by_age_file = f; break
        
        # Load decade_summary if exists
        for f in d.glob('decade_summary_n*_c*.csv'):
            try:
                decade_summaries.append(load_decade_summary(f, ds, short))
            except Exception as e:
                print(f"  Error loading decade_summary {f}: {e}")
            break
        
        # Load comparison, token frequency, and block summary from stats file
        if stats_file and stats_file.exists():
            try:
                comp_dfs.append(load_comparisons(stats_file, short))
                block_dfs.append(load_token_freq_block(stats_file, short))
                # Block summary (denominator per prefix_len)
                bs = load_block_summary(stats_file, short)
                bs['dataset'] = ds
                block_summaries.append(bs)
            except Exception as e:
                print(f"  Error loading {stats_file}: {e}")
        
        # Load by-age token frequencies
        if by_age_file and by_age_file.exists():
            try:
                dec_dfs.append(load_token_freq_decade(by_age_file, short))
            except Exception as e:
                print(f"  Error loading {by_age_file}: {e}")
        
        # Real counts (once per dataset)
        if real_plen is None:
            print(f"  Attempting to load real counts from {d.name}...")
            tc_dec = load_real_from_token_counts(d)
            if tc_dec:
                real_dec = tc_dec
                print(f"    Loaded decade counts from token_counts_by_decade file")
            
            orig, ages = d / 'original_sequences.parquet', d / 'ages.parquet'
            print(f"    Checking: orig={orig.exists()}, ages={ages.exists()}")
            if orig.exists() and ages.exists():
                real_plen, computed_dec = load_real_counts(orig, ages)
                print(f"    real_plen has {len(real_plen)} prefix_lens, computed_dec has {len(computed_dec)} decades")
                if real_plen:
                    # Show sample of what we got
                    sample_plen = list(real_plen.keys())[0] if real_plen else None
                    if sample_plen:
                        print(f"    Sample: prefix_len={sample_plen} has {len(real_plen[sample_plen])} token counts")
                if real_dec is None:
                    real_dec = computed_dec
                print(f"  Real counts loaded from {d.name}")
            else:
                print(f"    WARNING: Missing parquet files in {d.name}")
    
    # Export comparisons (long format: concat all experiments)
    if comp_dfs:
        m = pd.concat(comp_dfs, ignore_index=True)
        n = clean_comparisons(m, export_dir / f'comparisons_{ds}.csv')
        print(f"  comparisons_{ds}.csv: {n} rows")
    
    # Export blockwise token frequencies
    if block_dfs:
        print(f"\n  Merging {len(block_dfs)} block dataframes...")
        m = merge_all(block_dfs, ['prefix_len', 'token_id'])
        print(f"  After merge: {len(m)} rows, columns: {list(m.columns)}")
        
        # Debug: check real_plen status
        print(f"  real_plen is None: {real_plen is None}")
        if real_plen:
            print(f"  real_plen has {len(real_plen)} prefix_lens")
            m = add_real_block(m, real_plen)
            print(f"  After add_real_block: columns = {list(m.columns)}")
        else:
            print(f"  WARNING: No real_plen available for blockwise!")
        
        n = clean_token_freq(m, export_dir / f'blockwise_{ds}.csv', ['prefix_len', 'token_id'])
        print(f"  blockwise_{ds}.csv: {n} rows")
    else:
        print(f"\n  WARNING: No block dataframes collected for {ds}!")
    
    # Export decade token frequencies
    if dec_dfs:
        m = merge_all(dec_dfs, ['decade', 'token_id'])
        if real_dec:
            m = add_real_decade(m, real_dec)
        n = clean_token_freq(m, export_dir / f'decade_{ds}.csv', ['decade', 'token_id'])
        print(f"  decade_{ds}.csv: {n} rows")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True)
    p.add_argument('--export-dir', required=True)
    args = p.parse_args()
    
    out_dir, exp_dir = Path(args.output_dir), Path(args.export_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by dataset
    by_ds = defaultdict(list)
    for d in out_dir.iterdir():
        if d.is_dir() and d.name.startswith('exp_'):
            _, ds = parse_exp(d.name)
            if ds: by_ds[ds].append(d)
    
    print(f"Found {sum(len(v) for v in by_ds.values())} experiments across {len(by_ds)} datasets")
    print(f"Datasets: {sorted(by_ds.keys())}")
    
    # Collect decade summaries and block summaries across all datasets
    decade_summaries = []
    block_summaries = []
    
    for ds, dirs in sorted(by_ds.items()):
        export_dataset(ds, dirs, exp_dir, decade_summaries, block_summaries)
    
    # Save merged decade_summary
    if decade_summaries:
        merged = pd.concat(decade_summaries, ignore_index=True)
        # Reorder columns: dataset, experiment first
        cols = ['dataset', 'experiment'] + [c for c in merged.columns if c not in ['dataset', 'experiment']]
        merged = merged[cols]
        merged.to_csv(exp_dir / 'decade_summary.csv', index=False)
        print(f"\n  decade_summary.csv: {len(merged)} rows")
    
    # Save merged block_summary
    if block_summaries:
        merged = pd.concat(block_summaries, ignore_index=True)
        # Reorder columns: dataset, experiment, prefix_len, denominator
        merged = merged[['dataset', 'experiment', 'prefix_len', 'denominator']]
        merged = merged.sort_values(['dataset', 'experiment', 'prefix_len'])
        merged.to_csv(exp_dir / 'block_summary.csv', index=False)
        print(f"  block_summary.csv: {len(merged)} rows")
    
    print(f"\n=== Done: {exp_dir} ===")

if __name__ == '__main__':
    main()
