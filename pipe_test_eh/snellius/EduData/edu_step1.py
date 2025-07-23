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

    # Keep extra columns
    known = set(colmap.values())
    extra_cols = [c for c in df.columns if c not in known]

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

    # Merge birth date
    df = df.merge(bg_df, on="RINPERSOON", how="left")

    # Compute daysSinceFirst & age
    df["daysSinceFirst"] = (
        pd.to_datetime(df["event_date"]) - pd.to_datetime(GENESIS_DATE)
    ).dt.days

    df["age"] = (
        (pd.to_datetime(df["event_date"]) - pd.to_datetime(df["birth_date"]))
        .dt.days / 365.2425
    )

    # drop helper if requested
    df = df.drop(columns=["birth_date", "event_date"], errors="ignore")

    # Remove other id columns (except allowed)
    id_cols_to_drop = [
        c for c in df.columns
        if c.lower().endswith("id") and c not in ["RINPERSOON", "RINPERSOON2"]
    ]
    df = df.drop(columns=id_cols_to_drop, errors="ignore")

    # Write parquet
    data_out = out_dir / f"{name}.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), data_out)

    # Build meta
    specials_cfg = file_cfg.get("special_values", {})
    meta_rows = []
    for col in df.columns:
        s = df[col]
        col_type = infer_col_type(s)
        labels = specials_cfg.get(col, {})
        meta_rows.append({"Name": col, "Type": col_type, "ValueLabels": labels})

    meta_df = pd.DataFrame(meta_rows)
    meta_out = out_dir / f"{name}_meta.parquet"
    pq.write_table(pa.Table.from_pandas(meta_df, preserve_index=False), meta_out)

    logging.info(f"Done {name}: {data_out.name}, {meta_out.name}")

def stage_convert(cfg: dict):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load background
    bg_cfg = cfg["background"]
    bg = read_table(bg_cfg["path"], bg_cfg.get("format", "parquet"))
    rp = bg_cfg["columns"]["rinpersoon"]
    by = bg_cfg["columns"]["birth_year"]
    bm = bg_cfg["columns"]["birth_month"]
    assumed_day = bg_cfg.get("assumed_birth_day", 15)

    birth_dates = build_birth_dates(bg, by, bm, assumed_day)
    bg = bg[[rp]].rename(columns={rp: "RINPERSOON"})
    bg["birth_date"] = birth_dates

    # Process each input with a progress bar
    for fcfg in tqdm(cfg["inputs"], desc="Files"):
        process_one_file(fcfg, bg, out_dir)

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
        stage_convert(cfg)
    elif args.stage == "stats_corr":
        stage_stats_corr(cfg)

if __name__ == "__main__":
    main()
