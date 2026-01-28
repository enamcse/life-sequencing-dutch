#!/usr/bin/env python3
"""
Compact export for generative evaluation results.
Outputs per dataset:
1. blockwise_{dataset}.csv - Comparisons + Token frequencies by prefix_len
2. decade_{dataset}.csv - Token frequencies by decade

Usage: python export_for_plotting.py --output-dir /path/to/gen_eval --export-dir /path/to/export
"""
import argparse, os, re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import pyarrow.parquet as pq

def parse_exp(name):
    """Extract (short_name, dataset) from experiment folder name.
    
    Examples:
        exp_n100_c100_h20_g100_k20_t08_GenBASE_GD0 -> (exp_..._GenBASE, GD0)
        exp_n100_c100_h20_g100_k20_t08_GenBASEbd_GDB0 -> (exp_..._GenBASEbd, GDB0)
    
    Note: Birthday models have 'bd' in model name AND use GDB* datasets.
    If model has 'bd' but dataset is GD*, we infer GDB* dataset.
    """
    # Pattern: exp_n{n}_c{c}_h{h}_g{g}_k{k}_t{t}_{model}_{dataset}
    m = re.match(r'(exp_n\d+_c\d+_h\d+_g\d+_k\w+_t\d+_(\w+))_(GD\w*)$', name)
    if not m:
        return (name, None)
    
    short_name = m.group(1)
    model_part = m.group(2)  # e.g., GenBASE or GenBASEbd
    dataset = m.group(3)     # e.g., GD0 or GDB0
    
    # If model has 'bd' but dataset doesn't have 'B', fix it
    if 'bd' in model_part.lower() and not dataset.startswith('GDB'):
        # Convert GD0 -> GDB0, GD1 -> GDB1, etc.
        dataset = 'GDB' + dataset[2:]
    
    return (short_name, dataset)

def load_real_counts(orig_path, ages_path, h=20):
    """Compute real token counts from original_sequences."""
    try:
        orig = pq.read_table(orig_path).to_pandas()
        ages = pq.read_table(ages_path).to_pandas()
    except: return {}, {}
    
    orig = orig[~orig['is_buddy']]
    ages = ages[~ages['is_buddy']]
    
    plen_counts = defaultdict(lambda: defaultdict(int))
    dec_counts = defaultdict(lambda: defaultdict(int))
    plens = [7, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    for _, r in orig.iterrows():
        toks = list(map(int, r['original_sequence'].split(',')))
        rlen = r['real_length']
        ar = ages[ages['local_idx'] == r['local_idx']]
        if len(ar) == 0: continue
        ag = list(map(int, ar.iloc[0]['age_stream'].split(',')))
        
        for p in plens:
            if p + h > rlen: continue
            for i in range(p, min(p + h, rlen)):
                tid = toks[i]
                plen_counts[p][tid] += 1
                age = ag[p-1] if p-1 < len(ag) else ag[-1]
                dec_counts[f"{(age//10)*10}s"][tid] += 1
    
    return dict(plen_counts), dict(dec_counts)

def load_real_from_token_counts(d):
    """Load real counts from token_counts_by_decade_n*_c*.csv file."""
    # Find the file
    for f in d.glob('token_counts_by_decade_n*_c*.csv'):
        df = pd.read_csv(f)
        if 'decade' not in df.columns or 'token_id' not in df.columns:
            continue
        # Find real count column
        real_col = [c for c in df.columns if 'real' in c.lower()]
        if not real_col: continue
        
        # Build dict: decade -> {token_id -> count}
        result = defaultdict(dict)
        for _, row in df.iterrows():
            result[row['decade']][int(row['token_id'])] = int(row[real_col[0]])
        return dict(result)
    return None

def load_stats_block(path, suffix):
    """Load blockwise stats CSV - include BOTH comparisons and token_frequency rows."""
    df = pd.read_csv(path)
    
    # For comparisons (12 row types), we need: prefix_len, row_type, total_num, total_den
    # For token_frequency, we need: prefix_len, row_type, token_id, total_num, total_den
    
    comp_rows = df[df['row_type'] != 'token_frequency'][['prefix_len', 'row_type', 'total_num', 'total_den']].copy()
    comp_rows['token_id'] = ''  # Empty for comparisons
    
    tf_rows = df[df['row_type'] == 'token_frequency'][['prefix_len', 'row_type', 'token_id', 'total_num', 'total_den']].copy()
    tf_rows = tf_rows.dropna(subset=['token_id'])
    tf_rows['token_id'] = tf_rows['token_id'].astype(int).astype(str)
    
    result = pd.concat([comp_rows, tf_rows], ignore_index=True)
    return result.rename(columns={'total_num': f'num_{suffix}', 'total_den': f'den_{suffix}'})

def load_by_age_stats(path, suffix):
    """Load by-age stats CSV (no row_type column - only token frequencies)."""
    df = pd.read_csv(path)
    cols = ['decade', 'token_id', 'total_num', 'total_den']
    df = df[[c for c in cols if c in df.columns]].copy()
    df['token_id'] = df['token_id'].astype(int)
    return df.rename(columns={'total_num': f'num_{suffix}', 'total_den': f'den_{suffix}'})

def merge_all(dfs, key_cols):
    """Merge list of dataframes on key columns."""
    if not dfs: return pd.DataFrame()
    m = dfs[0]
    for df in dfs[1:]:
        m = m.merge(df, on=key_cols, how='outer')
    return m.fillna(0)

def add_real_block(merged, real_counts):
    """Add real_count column for blockwise (only for token_frequency rows)."""
    if not real_counts: return merged
    rows = []
    for plen, tcounts in real_counts.items():
        for tid, cnt in tcounts.items():
            rows.append({'prefix_len': plen, 'row_type': 'token_frequency', 'token_id': str(tid), 'real_count': cnt})
    if not rows: return merged
    rdf = pd.DataFrame(rows)
    return merged.merge(rdf, on=['prefix_len', 'row_type', 'token_id'], how='left').fillna(0)

def add_real_decade(merged, real_counts):
    """Add real_count column for decade stats."""
    if not real_counts: return merged
    rows = []
    for dec, tcounts in real_counts.items():
        for tid, cnt in tcounts.items():
            rows.append({'decade': dec, 'token_id': tid, 'real_count': cnt})
    if not rows: return merged
    rdf = pd.DataFrame(rows)
    return merged.merge(rdf, on=['decade', 'token_id'], how='outer').fillna(0)

def clean_and_save(df, num_cols, path, key_cols):
    """Remove zero/negative rows, convert to int, save."""
    num_only = [c for c in num_cols if c.startswith('num_')]
    
    # Set negative numerators (and corresponding denominators) to 0
    for nc in num_only:
        mask = df[nc] < 0
        df.loc[mask, nc] = 0
        dc = nc.replace('num_', 'den_')
        if dc in df.columns:
            df.loc[mask, dc] = 0
    
    # Remove rows where ALL numerators are 0 (only for token_frequency rows, keep comparisons)
    if num_only:
        if 'row_type' in df.columns:
            # Only drop token_frequency rows with all zeros, keep comparison rows
            is_tf = df['row_type'] == 'token_frequency'
            all_zero = (df[num_only] == 0).all(axis=1)
            drop_mask = is_tf & all_zero
            df = df[~drop_mask].copy()
            # Remove 'token_frequency' from row_type (replace with empty)
            df.loc[df['row_type'] == 'token_frequency', 'row_type'] = ''
        else:
            # No row_type column (decade stats) - drop all-zero rows
            all_zero = (df[num_only] == 0).all(axis=1)
            df = df[~all_zero].copy()
    
    for c in num_cols:
        df[c] = df[c].astype(int)
    
    # Sort and save
    df = df.sort_values(key_cols)
    df.to_csv(path, index=False)
    return len(df)

def export_dataset(ds, dirs, export_dir):
    print(f"\n=== {ds}: {len(dirs)} experiments ===")
    
    block_dfs, dec_dfs = [], []
    real_plen, real_dec = None, None
    
    for d in tqdm(dirs, desc=ds):
        short, _ = parse_exp(d.name)
        
        # Blockwise stats (statistics_n*_c*_summary.csv, not by_age)
        sp = None
        for f in d.glob('statistics_n*_c*_summary.csv'):
            if 'by_age' not in f.name:
                sp = f; break
        if sp and sp.exists():
            try:
                block_dfs.append(load_stats_block(sp, short))
            except Exception as e:
                print(f"  Error loading {sp}: {e}")
        
        # By-age stats (statistics_by_age_n*_c*_summary.csv)
        ap = None
        for f in d.glob('statistics_by_age_n*_c*_summary.csv'):
            ap = f; break
        if ap and ap.exists():
            try:
                dec_dfs.append(load_by_age_stats(ap, short))
            except Exception as e:
                print(f"  Error loading {ap}: {e}")
        
        # Real counts (once per dataset)
        if real_plen is None:
            # Try token_counts_by_decade file first
            tc_dec = load_real_from_token_counts(d)
            if tc_dec:
                real_dec = tc_dec
                print(f"  Real decade counts from token_counts_by_decade")
            
            # Compute from original sequences for blockwise
            orig, ages = d / 'original_sequences.parquet', d / 'ages.parquet'
            if orig.exists() and ages.exists():
                real_plen, computed_dec = load_real_counts(orig, ages)
                if real_dec is None:
                    real_dec = computed_dec
                print(f"  Real counts from {d.name}")
    
    # Export blockwise (includes 12 comparison rows + token frequencies)
    if block_dfs:
        m = merge_all(block_dfs, ['prefix_len', 'row_type', 'token_id'])
        if real_plen:
            m = add_real_block(m, real_plen)
        num_cols = [c for c in m.columns if c.startswith(('num_', 'den_', 'real_'))]
        n = clean_and_save(m, num_cols, export_dir / f'blockwise_{ds}.csv', ['prefix_len', 'row_type', 'token_id'])
        print(f"  blockwise_{ds}.csv: {n} rows")
    
    # Export decade (only token frequencies)
    if dec_dfs:
        m = merge_all(dec_dfs, ['decade', 'token_id'])
        if real_dec:
            m = add_real_decade(m, real_dec)
        num_cols = [c for c in m.columns if c.startswith(('num_', 'den_', 'real_'))]
        n = clean_and_save(m, num_cols, export_dir / f'decade_{ds}.csv', ['decade', 'token_id'])
        print(f"  decade_{ds}.csv: {n} rows")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True)
    p.add_argument('--export-dir', required=True)
    args = p.parse_args()
    
    out_dir, exp_dir = Path(args.output_dir), Path(args.export_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by dataset (with bd->GDB correction)
    by_ds = defaultdict(list)
    for d in out_dir.iterdir():
        if d.is_dir() and d.name.startswith('exp_'):
            _, ds = parse_exp(d.name)
            if ds: by_ds[ds].append(d)
    
    print(f"Found {sum(len(v) for v in by_ds.values())} experiments across {len(by_ds)} datasets")
    print(f"Datasets: {sorted(by_ds.keys())}")
    
    for ds, dirs in sorted(by_ds.items()):
        export_dataset(ds, dirs, exp_dir)
    
    print(f"\n=== Done: {exp_dir} ===")

if __name__ == '__main__':
    main()

