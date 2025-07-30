#!/usr/bin/env python3
import os, re, json, argparse
import pandas as pd
import logging
from tqdm import tqdm

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def main(cfg):
    node_dir       = cfg["node_dir"]
    pop_file       = cfg["pop_bucket_file"]
    id_col         = cfg["id_col"]
    year_col       = cfg["year_col"]
    bucket_col     = cfg["bucket_col"]
    output_col     = cfg["output_col"]
    pattern        = cfg["node_pattern"]
    meta_suffix    = cfg["meta_suffix"]

    # 1) load pop buckets
    pop = pd.read_parquet(pop_file)
    logging.info(f"Loaded population buckets from {pop_file} with {len(pop):,} rows")
    # rename for join
    pop = pop.rename(columns={bucket_col: output_col})[[id_col, year_col, output_col]]
    logging.info(f"Columns: {', '.join(pop.columns)}")

    # group by year for faster lookup
    pop_groups = {yr: grp.set_index(id_col)[output_col]
                  for yr, grp in pop.groupby(year_col)}

    # 2) process each node_features_YEAR.parquet
    for fname in tqdm(sorted(os.listdir(node_dir))):
        if not re.match(pattern.replace("*", r"\d{4}"), fname):
            continue
        year_match = re.search(r"(\d{4})", fname)
        if not year_match:
            logging.warning(f"cannot extract year from {fname}, skipping")
            continue
        yr = int(year_match.group(1))

        data_path = os.path.join(node_dir, fname)
        meta_path = os.path.join(
            node_dir,
            fname.replace(".parquet", meta_suffix)
        )
        logging.info(f"\n▶ Year {yr}: augmenting {fname}")

        df = pd.read_parquet(data_path)

        # slice pop for this year
        logging.info(f"   • loading population buckets for year {yr}")
        series = pop_groups.get(yr)
        if series is None:
            logging.warning(f"   ⚠️  no bucket data for year {yr}; filling NaN")
            df[output_col] = pd.NA
        else:
            # join on RINPERSOON
            df = df.set_index(id_col)
            df[output_col] = series.reindex(df.index)
            df = df.reset_index()

        # write back data
        df.to_parquet(data_path, index=False)
        logging.info(f"   ✓ wrote augmented data ({len(df):,} rows)")

        # --- update meta ---
        if os.path.exists(meta_path):
            meta = pd.read_parquet(meta_path)
            # append a row for cluster_id
            new_row = {
                "Name":        output_col,
                "Type":        "Numeric",
                "ValueLabels": {}
            }
            # avoid duplicates
            if output_col not in meta["Name"].values:
                meta = pd.concat([meta, pd.DataFrame([new_row])], ignore_index=True)
                logging.info(f" Meta update: {meta}")
                meta["ValueLabels"] = meta["ValueLabels"].apply(
                    lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v
                )
                meta.to_parquet(meta_path, index=False)
                logging.info(f"   ✓ wrote meta update ({output_col})")
            else:
                logging.info(f"   ℹ️  meta already contains {output_col}, skipping")
        else:
            logging.warning(f"   ⚠️  meta file not found at {meta_path}")

if __name__ == "__main__":
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True, help="Path to JSON config")
    args = p.parse_args()
    cfg = json.load(open(args.cfg))
    main(cfg)
