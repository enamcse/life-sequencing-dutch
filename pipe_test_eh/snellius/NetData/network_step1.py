#!/usr/bin/env python3
import os
import glob
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import pandas.api.types as pst

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_background(cfg):
    bg = cfg["BACKGROUND"]
    if bg["format"] == "csv":
        df = pd.read_csv(bg["path"])
    else:
        df = pd.read_parquet(bg["path"])
    df = df.rename(columns={k: v for k, v in bg["columns"].items()})
    # build birth_date
    df["birth_month"] = df["birth_month"].fillna(1).astype(int)
    df["birth_year"]  = df["birth_year"].astype(int)
    df["birth_day"]   = bg.get("assumed_birth_day", 1)

    offset = bg.get('birth_year_offset', 0)
    if offset != 0:
        df["birth_year"] += offset
        logging.info(f'Adjusted birth years: {df["birth_year"].min()} to {df["birth_year"].max()}')

    df["birth_date"]  = pd.to_datetime(dict(
        year = df["birth_year"],
        month= df["birth_month"],
        day  = df["birth_day"]
    ))
    return df[["RINPERSOON", "birth_date"]]

def convert(cfg):
    logging.info("Starting conversion process...")
    # 1) read raw network parquet(s)
    raw_dir = cfg["RAW_PARQ_DIR"]
    pattern = os.path.join(raw_dir, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files in {raw_dir}")
    dfs = [pd.read_parquet(fp) for fp in files]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    logging.info(f"Loaded {len(df):,} rows from {len(files)} files")

    # 2) merge background
    bg = load_background(cfg)
    logging.info(f"Loaded background data with {len(bg):,} rows")

    # align types
    df[cfg["ID_COLUMN"]] = df[cfg["ID_COLUMN"]].astype(str)
    bg[cfg["ID_COLUMN"]] = bg[cfg["ID_COLUMN"]].astype(str)
    df = df.merge(bg, on=cfg["ID_COLUMN"], how="inner")
    logging.info(f"Merged background data, now {len(df):,} rows")

    # 3) compute event_date, daysSinceFirstEvent, age
    ev = cfg["EVENT_DATE"]
    df["event_date"] = pd.to_datetime(dict(
        year = df[ev["year_column"]].astype(int),
        month= ev["month"],
        day  = ev["day"]
    ))
    genesis = pd.to_datetime(cfg["GENESIS_DATE"])

    logging.info(f"Calculating daysSinceFirstEvent and age...")
    df["daysSinceFirstEvent"] = (df["event_date"] - genesis).dt.days
    # fractional age in years
    df["age"] = ((df["event_date"] - df["birth_date"])
                 .dt.days / 365.2425).round(2)

    # 4) select final columns
    final_cols = [cfg["ID_COLUMN"], "daysSinceFirstEvent", "age"] + cfg["EVENT_COLUMNS"]
    df_final = df[final_cols]
    logging.info(f"Final columns: {final_cols}")

    # 5) write data parquet
    ensure_dir(cfg["OUTPUT_DIR"])
    out_data = os.path.join(cfg["OUTPUT_DIR"], cfg["DATA_FILE"])
    df_final.to_parquet(out_data, index=False)
    logging.info(f"Wrote data → {out_data}")

    # 6) build + write meta parquet
    logging.info("Building meta parquet...")
    rows = []

    for col in final_cols:
        typ = "Numeric" if pst.is_numeric_dtype(df_final[col].dtype) else "String"
        rows.append({
            "Name": col,
            "Type": typ,
            "ValueLabels": {}
        })
    meta_df = pd.DataFrame(rows)
    meta_df["ValueLabels"] = meta_df["ValueLabels"].apply(json.dumps)
    out_meta = os.path.join(cfg["OUTPUT_DIR"], cfg["META_FILE"])
    logging.info(f"Meta DataFrame: {meta_df}")
    meta_df.to_parquet(out_meta, index=False)
    logging.info(f"Wrote meta → {out_meta}")

if __name__ == "__main__":
    setup_logging()
    p = argparse.ArgumentParser(description="Step 1 for network data")
    p.add_argument("--cfg", required=True)
    args = p.parse_args()

    cfg = json.load(open(args.cfg))
    convert(cfg)
