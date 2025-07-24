#!/usr/bin/env python3
"""
Stages 0–2 for ZVW health cost data

Usage:
  python health_stage0_2_pipeline.py --cfg configs/health_stage0_2_cfg.json --stage convert
  python health_stage0_2_pipeline.py --cfg configs/health_stage0_2_cfg.json --stage stats_corr
  python health_stage0_2_pipeline.py --cfg configs/health_stage0_2_cfg.json --stage all
"""

import os
import re
import json
import glob
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pyreadstat
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr, pointbiserialr

# -------------------------- helpers ---------------------------

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def extract_year(filename, regex):
    m = re.search(regex, filename)
    if not m:
        raise ValueError(f"Cannot extract year from {filename} with regex {regex}")
    return int(m.group(0))

def phi_coefficient(x, y):
    # Pearson corr on binary vectors
    if x.sum() == 0 or y.sum() == 0:
        # Degenerate if no positives
        return np.nan
    return np.corrcoef(x, y)[0, 1]

def jaccard(x, y):
    inter = (x & y).sum()
    union = (x | y).sum()
    return inter / union if union else np.nan

def corr_matrix_heatmap(df, title, out_png, cluster=True, dpi=200, fs=9):
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    mat = df.values
    labels = df.index.tolist()

    if cluster and sns:
        g = sns.clustermap(df, cmap="coolwarm", center=0, linewidths=0.1,
                           figsize=(10, 10))
        plt.title(title)
        g.savefig(out_png, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.figure(figsize=(10, 8))
        plt.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90, fontsize=fs)
        plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=fs)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=dpi)
        plt.close()

# ----------------------- Stage 0 + 1 --------------------------

def stage_convert(cfg):
    raw_dir = cfg["RAW_SAV_DIR"]
    out_dir = cfg["PARTITIONED_OUT_DIR"]
    id_col  = cfg["ID_COLUMN"]
    year_regex = cfg["YEAR_EXTRACT_REGEX"]
    cost_cols = cfg["COST_COLUMNS"]
    fill_missing = cfg["FILL_MISSING_NUMERIC_WITH"]
    cast_int_to_float = cfg.get("CAST_INT_TO_FLOAT", True)

    ensure_dir(out_dir)

    # global list ensures consistent schema
    all_cols_set = set([id_col] + cost_cols)
    if cfg.get("SECOND_ID_COLUMN"): all_cols_set.add(cfg["SECOND_ID_COLUMN"])

    sav_files = sorted(glob.glob(os.path.join(raw_dir, "*.sav")))
    if not sav_files:
        raise FileNotFoundError(f"No .sav files found in {raw_dir}")

    for sav in sav_files:
        year = extract_year(os.path.basename(sav), year_regex)
        print(f"Reading {sav} (year {year})")
        df, meta = pyreadstat.read_sav(sav, apply_value_formats=False)

        # Ensure ID column(s) exist
        if id_col not in df.columns:
            # try to locate any RIN prefix
            candidates = [c for c in df.columns if c.upper().startswith("RIN")]
            if candidates:
                df = df.rename(columns={candidates[0]: id_col})
            else:
                raise KeyError(f"{id_col} not found in {sav}")

        if cfg.get("SECOND_ID_COLUMN") and cfg["SECOND_ID_COLUMN"] not in df.columns:
            # if absent, we'll add later

            pass

        # Keep only desired columns that exist
        keep = [c for c in all_cols_set if c in df.columns]
        df = df[keep].copy()
        df["year"] = year

        # Add missing cost columns as NaN (or fill value)
        missing_cols = [c for c in all_cols_set if c not in df.columns and c != "year"]
        for mc in missing_cols:
            df[mc] = np.nan

        # Enforce numeric dtypes to be float (for NaN compatibility)
        if cast_int_to_float:
            for c in cost_cols:
                if c in df.columns and pd.api.types.is_integer_dtype(df[c]):
                    df[c] = df[c].astype("float64")

        # Fill numeric missing if configured
        if fill_missing is not None:
            for c in cost_cols:
                if c in df.columns:
                    df[c] = df[c].fillna(fill_missing)

        # Write as partitioned by year
        year_dir = os.path.join(out_dir, f"year={year}")
        ensure_dir(year_dir)
        out_file = os.path.join(year_dir, f"part-{year}.parquet")
        df.to_parquet(out_file, index=False)
        print(f"  -> wrote {out_file}")

    print("Stage convert done.")

# ----------------------- Stage 2 ------------------------------

def stage_stats_corr(cfg):
    part_dir = cfg["PARTITIONED_OUT_DIR"]
    out_dir = cfg["OUTPUT_DIR"]
    ensure_dir(out_dir)

    id_col = cfg["ID_COLUMN"]
    second_id = cfg.get("SECOND_ID_COLUMN")
    cost_cols = cfg["COST_COLUMNS"]

    corr_cfg = cfg["CORR_SETTINGS"]
    vis_cfg  = cfg["VISUALIZATION"]

    # -------- load all parquet parts lazily ----------
    print("Reading partitioned parquet dataset (no hive partitioning)...")
    # Build dataset
    dataset = pq.ParquetDataset(part_dir, partitioning=None)
    table = dataset.read()  # With 500GB RAM might still be fine; if not, switch to scan->to_pandas in chunks
    df = table.to_pandas()

    # Sanity: ensure all columns present
    for c in cost_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Basic per-column stats (zeros, non-zeros, missing etc.)
    stats_rows = []
    total_rows = len(df)
    for c in cost_cols:
        s = df[c]
        is_na = s.isna()
        is_zero = s.fillna(0) == 0
        nz = (~is_na) & (~is_zero)
        stats_rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "total_rows": total_rows,
            "na_rows": int(is_na.sum()),
            "zero_rows": int(is_zero.sum()),
            "nonzero_rows": int(nz.sum()),
            "pct_value_exists": float(100 * (~is_na).sum() / total_rows),
            "sum": float(s.sum(skipna=True)),
            "mean": float(s.mean(skipna=True)),
            "median": float(s.median(skipna=True)),
            "std": float(s.std(skipna=True)),
            "min": float(s.min(skipna=True)) if (~is_na).any() else np.nan,
            "max": float(s.max(skipna=True)) if (~is_na).any() else np.nan
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_csv = os.path.join(out_dir, "column_stats_recomputed.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"Wrote stats: {stats_csv}")

    # ---------- Correlation calculations -------------
    # Presence matrix
    presence = pd.DataFrame({
        c: (df[c].fillna(0) > 0).astype(np.int8) for c in cost_cols
    })

    # Value matrix for non-zeros pairs: we will do pairwise, so no need to precompute big matrix
    # but we can log-transform entire df once
    if corr_cfg["log_transform"] == "log1p":
        log_df = np.log1p(df[cost_cols].fillna(0))
    else:
        log_df = df[cost_cols].fillna(0)

    pairs = []
    n_cols = len(cost_cols)
    for i in range(n_cols):
        a = cost_cols[i]
        for j in range(i+1, n_cols):
            b = cost_cols[j]

            # rows where both are not NaN
            mask_avail = df[[a, b]].notna().all(axis=1)
            subA = df.loc[mask_avail, a]
            subB = df.loc[mask_avail, b]
            n_all = mask_avail.sum()

            # presence-presence
            pa = presence.loc[mask_avail, a].values
            pb = presence.loc[mask_avail, b].values
            phi = phi_coefficient(pa, pb) if n_all >= corr_cfg["min_all_pairs"] else np.nan
            jac = jaccard(pa, pb) if n_all >= corr_cfg["min_all_pairs"] else np.nan

            # value-value on non-zero pairs
            nz_mask = (subA > 0) & (subB > 0)
            n_nz = nz_mask.sum()
            if n_nz >= corr_cfg["min_nonzero_pairs"]:
                va = log_df.loc[mask_avail & nz_mask, a].values
                vb = log_df.loc[mask_avail & nz_mask, b].values
                pear = float(np.corrcoef(va, vb)[0, 1]) if "pearson" in corr_cfg["value_metrics"] else np.nan
                spear = float(spearmanr(va, vb).correlation) if "spearman" in corr_cfg["value_metrics"] else np.nan
            else:
                pear = spear = np.nan

            # value vs presence (point-biserial)
            if n_all >= corr_cfg["min_all_pairs"]:
                pbis_a = float(pointbiserialr(pa, subB.fillna(0)).correlation)
                pbis_b = float(pointbiserialr(pb, subA.fillna(0)).correlation)
            else:
                pbis_a = pbis_b = np.nan

            pairs.append({
                "A": a, "B": b,
                "n_all": int(n_all), "n_nonzero_pairs": int(n_nz),
                "phi": phi, "jaccard": jac,
                "pearson_log": pear, "spearman_log": spear,
                "pbis_A_presence_vs_B_value": pbis_a,
                "pbis_B_presence_vs_A_value": pbis_b
            })

    pairs_df = pd.DataFrame(pairs)
    pairs_csv = os.path.join(out_dir, "pairwise_correlations.csv")
    pairs_df.to_csv(pairs_csv, index=False)
    print(f"Wrote pairwise correlations: {pairs_csv}")

    # ----- Build square matrices for heatmaps (phi & pearson) -----
    # Presence phi
    phi_mat = pd.DataFrame(np.eye(n_cols), index=cost_cols, columns=cost_cols, dtype=float)
    pear_mat = pd.DataFrame(np.eye(n_cols), index=cost_cols, columns=cost_cols, dtype=float)

    for _, row in pairs_df.iterrows():
        a, b = row["A"], row["B"]
        phi_mat.loc[a, b] = phi_mat.loc[b, a] = row["phi"]
        pear_mat.loc[a, b] = pear_mat.loc[b, a] = row["pearson_log"]

    phi_csv = os.path.join(out_dir, "phi_matrix.csv")
    pear_csv = os.path.join(out_dir, "pearson_log_matrix.csv")
    phi_mat.to_csv(phi_csv)
    pear_mat.to_csv(pear_csv)
    print(f"Wrote matrices: {phi_csv}, {pear_csv}")

    if vis_cfg["heatmaps"]:
        ensure_dir(os.path.join(out_dir, "plots"))
        phi_png  = os.path.join(out_dir, "plots/phi_heatmap.png")
        pear_png = os.path.join(out_dir, "plots/pearson_log_heatmap.png")
        corr_matrix_heatmap(phi_mat, "Presence (Phi) Correlation", phi_png,
                            cluster=vis_cfg["cluster_order"],
                            dpi=vis_cfg["dpi"], fs=vis_cfg["font_size"])
        corr_matrix_heatmap(pear_mat, "Value-Value (log1p) Pearson", pear_png,
                            cluster=vis_cfg["cluster_order"],
                            dpi=vis_cfg["dpi"], fs=vis_cfg["font_size"])
        print(f"Wrote heatmaps: {phi_png}, {pear_png}")

    print("Stage stats_corr done.")

# --------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--stage", choices=["convert","stats_corr","all"], default="all")
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = json.load(f)

    np.random.seed(cfg.get("RANDOM_SEED", 42))

    if args.stage in ("convert","all"):
        stage_convert(cfg)

    if args.stage in ("stats_corr","all"):
        stage_stats_corr(cfg)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
