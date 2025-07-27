#!/usr/bin/env python3
import logging
import os, json, glob, argparse
from pathlib import Path
import numpy  as np
import pandas as pd
from tqdm import tqdm


# ---------- helpers ----------

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
    # rename to canonical
    df = df.rename(columns={k: v for k, v in bg["columns"].items()})
    # build full birth_date
    df["birth_month"] = df["birth_month"].fillna(1).astype(int)
    df["birth_year"]  = df["birth_year"].astype(int)
    df["birth_day"]   = bg.get("assumed_birth_day", 1)

    offset = bg.get('birth_year_offset', 0)
    logging.info(f'Adjusting birth year offset = {offset} years...')

    if offset != 0:
        df["birth_year"] += offset
        logging.info(f'Adjusted birth years: {df["birth_year"].min()} to {df["birth_year"].max()}')

    df["birth_date"]  = pd.to_datetime(dict(
        year=df["birth_year"],
        month=df["birth_month"],
        day=df["birth_day"]
    ))
    return df[["RINPERSOON","birth_date"]]

def compute_event_df(cfg):
    # read all partitioned parquet parts
    parts = glob.glob(os.path.join(cfg["RAW_PARQ_DIR"], "year=*", "*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts in {cfg['RAW_PARQ_DIR']}")
    dfs = [pd.read_parquet(p) for p in sorted(parts)]
    df = pd.concat(dfs, ignore_index=True)
    return df

def convert(cfg):
    # 1) load raw events
    df = compute_event_df(cfg)

    # 2) merge in background to get birth_date
    bg = load_background(cfg)
    df = df.merge(bg, on="RINPERSOON", how="left")

    # 3) compute event_date, daysSinceFirst, age
    ev = cfg["EVENT_DATE"]
    df["event_date"] = pd.to_datetime(dict(
        year = df[ev["year_column"]].astype(int),
        month = ev["month"],
        day   = ev["day"]
    ))
    genesis = pd.to_datetime(cfg["GENESIS_DATE"])
    df["daysSinceFirst"] = (df["event_date"] - genesis).dt.days
    df["age"] = (
        df["event_date"].dt.year - df["birth_date"].dt.year
        - ((df["event_date"].dt.month  < df["birth_date"].dt.month) |
           (df["event_date"].dt.month == df["birth_date"].dt.month) &
           (df["event_date"].dt.day   < df["birth_date"].dt.day)
          ).astype(int)
    )

    # 4) aggregate cost‑groups
    for group, cols in tqdm(cfg["COST_GROUPS"].items(), desc="Aggregating cost groups"):
        # ensure missing raw cols exist
        for c in tqdm(cols, desc=f"Processing columns for {group}", leave=False):
            if c not in df.columns:
                df[c] = 0.0
        df[group] = df[cols].sum(axis=1)

    # 5) drop pure zero rows if requested
    if cfg.get("DROP_ZERO_ROWS", False):
        group_cols = list(cfg["COST_GROUPS"].keys())
        df = df[df[group_cols].sum(axis=1) != 0]

    # 6) select final columns
    out_cols = [
        cfg["ID_COLUMN"],
        "daysSinceFirst",
        "age"
    ] + list(cfg["COST_GROUPS"].keys())

    df_final = df[out_cols]

    # 7) write a.parquet
    ensure_dir(cfg["OUTPUT_DIR"])
    out_parq = os.path.join(cfg["OUTPUT_DIR"], cfg["DATA_FILE"])
    df_final.to_parquet(out_parq, index=False)
    logging.info(f"Wrote data → {out_parq}")

    # 8) build and write meta.parquet
    rows = []
    for col in out_cols:
        typ = "Numeric" if np.issubdtype(df_final[col].dtype, np.number) else "String"
        rows.append({
            "Name": col,
            "Type": typ,
            "ValueLabels": {}
        })
    meta_df = pd.DataFrame(rows)
    # ensure ValueLabels column is a dict
    meta_df["ValueLabels"] = meta_df["ValueLabels"].apply(json.dumps)
    out_meta = os.path.join(cfg["OUTPUT_DIR"], cfg["META_FILE"])
    meta_df.to_parquet(out_meta, index=False)
    logging.info(f"Wrote meta → {out_meta}")

if __name__ == "__main__":
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",    required=True)
    p.add_argument("--stage",  choices=["convert"], default="convert")
    args = p.parse_args()

    cfg = json.load(open(args.cfg))
    if args.stage == "convert":
        convert(cfg)
