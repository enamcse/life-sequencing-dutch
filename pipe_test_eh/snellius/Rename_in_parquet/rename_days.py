#!/usr/bin/env python3
"""
Rename `daysSinceFirst` → `daysSinceFirstEvent`
in all Parquet data files and their corresponding _meta.parquet files.

Usage:
  python rename_days.py --data-dir /path/to/parquets \
                       --meta-dir /path/to/meta_parquets
"""

import os
import argparse
import pandas as pd

OLD = "daysSinceFirst"
NEW = "daysSinceFirstEvent"

def rename_in_data(data_dir):
    for fname in os.listdir(data_dir):
        if not fname.endswith(".parquet"):
            continue
        path = os.path.join(data_dir, fname)
        df = pd.read_parquet(path)
        if OLD in df.columns:
            df = df.rename(columns={OLD: NEW})
            df.to_parquet(path, index=False)
            print(f"Updated data file: {fname}")

def rename_in_meta(meta_dir):
    for fname in os.listdir(meta_dir):
        if not fname.endswith("_meta.parquet"):
            continue
        path = os.path.join(meta_dir, fname)
        df = pd.read_parquet(path)
        if OLD in df["Name"].values:
            df.loc[df["Name"] == OLD, "Name"] = NEW
            df.to_parquet(path, index=False)
            print(f"Updated meta file: {fname}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="Directory containing your *.parquet data files")
    p.add_argument("--meta-dir", required=True,
                   help="Directory containing your *_meta.parquet files")
    args = p.parse_args()

    print("Renaming in data files…")
    rename_in_data(args.data_dir)
    print("Renaming in meta files…")
    rename_in_meta(args.meta_dir)
    print("Done.")

if __name__ == "__main__":
    main()
