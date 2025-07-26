#!/usr/bin/env python3
"""
Convert raw parquet/csv education files (CITO & CE) to Step-1 format.

Usage:
    python edu_step1_pipeline.py --cfg edu_step1_cfg.json --stage convert
"""

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

GENESIS_DATE = dt.date(1971, 12, 30)

# ---------- helpers ----------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def read_table(path: str, fmt: str) -> pd.DataFrame:
    logging.info(f'Loading {fmt} file from {path}')
    if fmt.lower() == "parquet":
        return pd.read_parquet(path)
    elif fmt.lower() == "csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format '{fmt}' for {path}")

def build_birth_dates(df_bg: pd.DataFrame, year_col: str, month_col: str, day: int) -> pd.Series:
    return pd.to_datetime(
        dict(year=df_bg[year_col], month=df_bg[month_col], day=day),
        errors="coerce"
    ).dt.date

def infer_col_type(series: pd.Series) -> str:
    return "Numeric" if pd.api.types.is_numeric_dtype(series) else "String"

def process_one_file(file_cfg: dict, bg_df: pd.DataFrame, out_dir: Path):
    name = file_cfg["name"]
    logging.info(f"Processing {name}")

    df = read_table(file_cfg["path"], file_cfg.get("format", "parquet"))

    # rename main columns
    colmap = {
        file_cfg["columns"]["rinpersoon"]: "RINPERSOON",
        file_cfg["columns"]["year"]: "year",
        file_cfg["columns"]["assessment"]: "assessment",
        file_cfg["columns"]["value"]: "value"
    }
    df = df.rename(columns=colmap)
    df["RINPERSOON"] = pd.to_numeric(df["RINPERSOON"], errors="coerce").astype("Int64") 


    # Keep extra columns
    known = set(colmap.values())
    extra_cols = [c for c in df.columns if c not in known]
    if extra_cols != []:
        logging.info(f"Found extra columns: {extra_cols}") 

    logging.info('Creating Event date...')
    # Event date
    dcfg = file_cfg.get("date", {"source": "year", "month": 5, "day": 1})
    event_dates = pd.to_datetime(
        dict(
            year=df[dcfg["source"]],
            month=dcfg.get("month", 5),
            day=dcfg.get("day", 1)
        ), errors="coerce"
    ).dt.date
    df["event_date"] = event_dates  # drop later if you don't want
    logging.info('Event date created for all.')

    logging.info('RINPERSOON sanity checking before merging with background file...')
    # --- dtype sanity just before merging ---
    if df["RINPERSOON"].dtype != bg_df["RINPERSOON"].dtype:
        logging.warning(
            "RINPERSOON dtype mismatch (%s vs %s); coercing both to Int64",
            df["RINPERSOON"].dtype, bg_df["RINPERSOON"].dtype
        )
        df["RINPERSOON"] = pd.to_numeric(df["RINPERSOON"], errors="coerce").astype("Int64")
        bg_df["RINPERSOON"] = pd.to_numeric(bg_df["RINPERSOON"], errors="coerce").astype("Int64")
    logging.info('RINPERSOON sanity check completed.')

    logging.info(f'Start merging... len(df) = {len(df)}')
    # Merge birth date
    df = df.merge(bg_df, on="RINPERSOON", how="left")
    logging.info(f'Merged background file. len(df) = {len(df)}')

    
    logging.info(f'{df[["RINPERSOON", "event_date", "birth_date"]].head(10)}')
    logging.info(f'Missing event_date # {df["event_date"].isna().sum()}')
    logging.info(f'Missing birth_date # {df["birth_date"].isna().sum()}')

    logging.info('Start Calculating daysSinceFirst and age...')
    # Compute daysSinceFirst & age
    df["daysSinceFirst"] = (
        pd.to_datetime(df["event_date"]) - pd.to_datetime(GENESIS_DATE)
    ).dt.days

    df["age"] = (
        (pd.to_datetime(df["event_date"]) - pd.to_datetime(df["birth_date"]))
        .dt.days / 365.2425
    )
    logging.info(f"Calculation Completed.\nFound {df['age'].nunique()} unique daysSinceFirst and {df['age'].nunique()} unique age.")

    logging.info('Dropping birth_date and event_date....')
    # drop helper if requested
    df = df.drop(columns=["birth_date", "event_date"], errors="ignore")


    # Remove other id columns (except allowed)
    id_cols_to_drop = [
        c for c in df.columns
        if c.lower().endswith("id") and c not in ["RINPERSOON", "RINPERSOON2"]
    ]
    logging.info(f'Dropping other columns: {id_cols_to_drop}...')
    df = df.drop(columns=id_cols_to_drop, errors="ignore")

    # Write parquet
    data_out = out_dir / f"{name}.parquet"
    logging.info(f'Writing parquet file: {data_out}')
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), data_out)
    logging.info(f'Parquet file writing completed.')

    logging.info('Building meta file...')
    # Build meta
    specials_cfg = file_cfg.get("special_values", {})
    meta_rows = []
    for col in df.columns:
        s = df[col]
        col_type = infer_col_type(s)
        labels = specials_cfg.get(col, {})
        meta_rows.append({"Name": col, "Type": col_type, "ValueLabels": labels})

    meta_df = pd.DataFrame(meta_rows)

    # Serializable ValueLabels dict -> JSON string so Parquet can write it
    meta_df['ValueLabels'] = meta_df['ValueLabels'].apply(json.dumps)

    meta_out = out_dir / f"{name}_meta.parquet"
    logging.info(f'Writing meta file: {meta_out}')
    pq.write_table(pa.Table.from_pandas(meta_df, preserve_index=False), meta_out)

    logging.info(f"Done {name}: {data_out.name}, {meta_out.name}")

def stage_convert(cfg: dict):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load background
    bg_cfg = cfg["background"]
    logging.info(f"Loading background file...")
    bg = read_table(bg_cfg["path"], bg_cfg.get("format", "parquet"))
    rp = bg_cfg["columns"]["rinpersoon"]
    by = bg_cfg["columns"]["birth_year"]
    bm = bg_cfg["columns"]["birth_month"]
    assumed_day = bg_cfg.get("assumed_birth_day", 15)

    logging.info(f'Background head: {bg.head(5)}')
    logging.info(f'Columns: {bg.columns.tolist()}')
    logging.info(f'Uniques year: {bg[by].unique()[:10]}')
    logging.info(f'Uniques month: {bg[bm].unique()[:10]}')

    offset = bg_cfg.get('birth_year_offset', 0)
    logging.info(f'Adjusting birth year offset = {offset} years...')

    if offset:
        bg[by] = bg[by] + offset
    

    birth_dates = build_birth_dates(bg, by, bm, assumed_day)
    logging.info('Renaming RINPERSOON column if named otherwise...')
    bg = bg[[rp]].rename(columns={rp: "RINPERSOON"})
    bg["RINPERSOON"] = pd.to_numeric(bg["RINPERSOON"], errors="coerce").astype("Int64")  
    bg["birth_date"] = birth_dates
    logging.info('Background data fully loaded.')
    
    logging.info('Start Processing input files...')
    # Process each input with a progress bar
    for fcfg in tqdm(cfg["inputs"], desc="Files"):
        process_one_file(fcfg, bg, out_dir)
    logging.info('Processed all input files.')

def stage_stats_corr(cfg: dict):
    # Placeholder for your Stage 2; implement as needed.
    logging.info("Stage 'stats_corr' not implemented here.")

def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--stage", default="convert", choices=["convert", "stats_corr"])
    args = ap.parse_args()

    with open(args.cfg) as f:
        cfg = json.load(f)

    if args.stage == "convert":
        logging.info('Calling stage_convert...')
        stage_convert(cfg)
    elif args.stage == "stats_corr":
        logging.info('Calling stage_stats_corr...')
        stage_stats_corr(cfg)

if __name__ == "__main__":
    main()
