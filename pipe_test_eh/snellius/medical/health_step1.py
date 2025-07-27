#!/usr/bin/env python3
import logging
import os, json, glob, argparse, re
from pathlib import Path
import numpy  as np
import pandas as pd
from tqdm import tqdm
import pandas.api.types as pst
import pyreadstat


# ---------- helpers ----------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def extract_year(fname, regex):
    m = re.search(regex, fname)
    if not m:
        raise ValueError(f"Cannot extract year from '{fname}'")
    return int(m.group(0))
def load_background(cfg):
    bg = cfg["BACKGROUND"]
    if bg["format"] == "csv":
        df = pd.read_csv(bg["path"])
    else:
        df = pd.read_parquet(bg["path"])
    # rename to canonical
    df = df.rename(columns={k: v for k, v in bg["columns"].items()})
    logging.info(f'Loaded background data with {len(df):,} rows')
    logging.info(f'Columns: {", ".join(df.columns)}')
    logging.info(f'Sample:\n{df.head()}')
    # build full birth_date
    df["birth_month"] = df[bg['columns']["birth_month"]].fillna(1).astype(int)
    df["birth_year"]  = df[bg['columns']["birth_year"]].astype(int)
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
    raw_dir = cfg["RAW_SAV_DIR"]
    out_dir = cfg["OUTPUT_DIR"]
    sentinel = cfg["MISSING_VALUE_SENTINEL"]
    year_re  = cfg["YEAR_EXTRACT_REGEX"]
    cost_groups = cfg["COST_GROUPS"]
    id_col = cfg["ID_COLUMN"]

    # 1) read & concatenate all .sav
    frames = []
    savs = sorted(glob.glob(os.path.join(raw_dir, "*.sav")))
    if not savs:
        raise FileNotFoundError(f"No SAV files in {raw_dir}")
    for sav in savs:
        year = extract_year(os.path.basename(sav), year_re)
        print(f"⏳ Loading {sav} (year={year})")
        df, meta = pyreadstat.read_sav(sav, apply_value_formats=False)

        # ensure ID present
        if id_col not in df.columns:
            cands = [c for c in df.columns if c.upper().startswith("RIN")]
            if cands:
                df = df.rename(columns={cands[0]: id_col})
            else:
                raise KeyError(f"{id_col} not in {sav}")

        # extract year
        df["year"] = year

        # map any NaN → sentinel
        df = df.fillna(sentinel)

        # keep only id, year, and all cost cols that appear
        cols = [id_col, "year"]
        for grp in cost_groups.values():
            cols += [c for c in grp if c in df.columns]
        frames.append(df[cols])

    df = pd.concat(frames, ignore_index=True)
    frames.clear()

    # 2) merge background
    bg = load_background(cfg)
    logging.info(f"Loaded background data with {len(bg):,} rows")
    
    logging.info(f"Merging background data with {len(df):,} rows")

    # align types
    df[id_col] = df[id_col].astype(str)
    bg[id_col] = bg[id_col].astype(str)
    df = df.merge(bg, on=id_col, how="inner")
    logging.info(f"After merge: {len(df):,} rows")

    # 3) compute event_date, daysSinceFirst, age
    ev = cfg["EVENT_DATE"]
    df["event_date"] = pd.to_datetime(dict(
        year = df[ev["year_column"]].astype(int),
        month= ev["month"],
        day  = ev["day"]
    ))
    genesis = pd.to_datetime(cfg["GENESIS_DATE"])
    df["daysSinceFirst"] = (df["event_date"] - genesis).dt.days
    # age in years, floor
    df["age"] = (
        (df["event_date"] - df["birth_date"])
        .dt.days
        .div(365.2425)
    )

    # 4) aggregate cost‑groups
    logging.info("🔢 Aggregating cost groups:")
    for group, cols in tqdm(cfg["COST_GROUPS"].items(), desc="Aggregating cost groups"):
        logging.info(f"   • {group}")

        # (1) find which of these raw cols actually appeared in this DF
        present = [c for c in cols if c in df.columns]

        if not present:
            # NONE of these ever existed → truly unavailable
            df[group] = sentinel
        else:
            # (2) for the ones that do exist, treat
            #     NaN  as 0 (no claim) *not* as missing,
            #     sentinel as missing/unavailable if you like
            part = df[present].fillna(0.0)
            df[group] = part.sum(axis=1)

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
        # use pandas’ dtype check so we handle extension dtypes like Int64
        typ = "Numeric" if pst.is_numeric_dtype(df_final[col].dtype) else "String"
        # record sentinel in ValueLabels if a cost group
        labels = {}
        if col in cost_groups and sentinel is not None:
            labels = {str(sentinel): "missing or unavailable"}
        rows.append({"Name": col, "Type": typ, "ValueLabels": labels})
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
