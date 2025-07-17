import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json

def compute_stats(df, name, output_dir):
    group = df.groupby("year")["value"]
    stats = group.agg(["count", "max", "min", "median", "mean", "sum", "std"]).reset_index()
    total = df["value"].agg(["count", "max", "min", "median", "mean", "sum", "std"]).to_frame().T
    total.insert(0, "year", "ALL")
    summary = pd.concat([stats, total], ignore_index=True)
    summary.to_csv(os.path.join(output_dir, f"{name}_yearly_statistics.csv"), index=False)
    return summary

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    cito = pd.read_parquet(args.cito_file)
    ce = pd.read_parquet(args.ce_file)

    compute_stats(cito, "cito", args.output_dir)
    compute_stats(ce, "ce", args.output_dir)

    report = {
        "unique_rinpersson_cito": cito["rinpersson"].nunique(),
        "unique_rinpersson_ce": ce["rinpersson"].nunique(),
        "common_rinpersson": len(set(cito["rinpersson"]) & set(ce["rinpersson"])),
        "unique_assessment_cito": sorted(cito["assessment"].unique().tolist()),
        "unique_assessment_ce": sorted(ce["assessment"].unique().tolist())
    }

    with open(os.path.join(args.output_dir, "score_summary_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    plt.hist(cito["value"], bins=50, alpha=0.7, label="CITO")
    plt.title("CITO Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "cito_distribution.png"))
    plt.clf()

    plt.hist(ce["value"], bins=50, alpha=0.7, label="CE")
    plt.title("CE Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "ce_distribution.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cito_file", type=str, required=True)
    parser.add_argument("--ce_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
