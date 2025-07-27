#!/usr/bin/env python3
import os
import glob
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import pyreadstat
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def process_dataset(raw_dir, out_dir, name, spec):
    logging.info(f"\n▶ Processing '{name}'")
    pattern = os.path.join(raw_dir, spec["input_pattern"])
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for dataset '{name}' with pattern {pattern}")

    dfs = []
    for fpath in files:
        logging.info(f"   • loading {os.path.basename(fpath)}")
        if spec["format"] == "csv":
            df = pd.read_csv(fpath)
        elif spec["format"] == "space":
            # space‐delimited, headers quoted
            df = pd.read_csv(fpath, delim_whitespace=True, quotechar='"')
        else:
            raise ValueError(f"Unknown format '{spec['format']}' for {name}")
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    # keep only the configured columns
    df = df[spec["columns"]].copy()

    # rename daysSinceFirstEvent → daysSinceFirst
    if "daysSinceFirstEvent" in df.columns:
        df = df.rename(columns={"daysSinceFirstEvent": "daysSinceFirst"})

    # enforce dtypes
    for col in spec.get("numeric_cols", []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in spec.get("string_cols", []):
        if col in df.columns:
            df[col] = df[col].astype(str)

    # write the data parquet
    out_data = os.path.join(out_dir, f"{name}.parquet")
    df.to_parquet(out_data, index=False)
    logging.info(f"   ✔ Wrote {out_data} ({len(df):,} rows)")

    # ---- now build the meta parquet based on the SAV ----
    sav_path = os.path.join(raw_dir, spec.get("meta_sav", ""))
    rows = []
    if os.path.isfile(sav_path):
        logging.info(f"   • loading metadata from {sav_path}")
        _, meta = pyreadstat.read_sav(sav_path, apply_value_formats=False)

        # for every column in the final parquet
        for col in df.columns:
            # decide type
            typ = "Numeric" if col in spec.get("numeric_cols",[]) else "String"

            # grab SPSS value labels
            vl = meta.value_labels.get(col, {})

            # grab user‑defined missing values & ranges
            mv = meta.missing_user_values.get(col, [])[:]
            for lo, hi in meta.missing_ranges.get(col, []):
                mv.extend(list(range(lo, hi+1)))
            # mark all missing codes in the same dict
            missing_labels = {int(c): "system missing" for c in mv}

            # merge them (labels take precedence)
            all_labels = {**missing_labels, **{int(k):v for k,v in vl.items()}}

            rows.append({
                "Name": col,
                "Type": typ,
                "ValueLabels": all_labels
            })
    else:
        logging.warning(f"   ⚠️  No SAV found for metadata at {sav_path}; using empty ValueLabels")
        for col in df.columns:
            typ = "Numeric" if col in spec.get("numeric_cols",[]) else "String"
            rows.append({"Name":col, "Type":typ, "ValueLabels":{}})

    meta_df = pd.DataFrame(rows)
    # JSON‑serialize the dict in each row
    meta_df["ValueLabels"] = meta_df["ValueLabels"].apply(json.dumps, ensure_ascii=False)

    out_meta = os.path.join(out_dir, f"{name}_meta.parquet")
    meta_df.to_parquet(out_meta, index=False)
    logging.info(f"   ✔ Wrote {name}_meta.parquet")


def main():
    setup_logging()
    p = argparse.ArgumentParser(description="Convert Lucas health files to Parquet+meta")
    p.add_argument("--cfg", required=True, help="Path to JSON config")
    args = p.parse_args()

    cfg = json.load(open(args.cfg))
    raw_dir = cfg["RAW_DIR"]
    out_dir = cfg["OUTPUT_DIR"]
    ensure_dir(out_dir)

    for ds_name, spec in cfg["DATASETS"].items():
        process_dataset(raw_dir, out_dir, ds_name, spec)

if __name__ == "__main__":
    main()
