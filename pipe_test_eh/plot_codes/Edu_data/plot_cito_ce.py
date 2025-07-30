import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths and settings
cito_path = "cito_scores.parquet"
ce_path = "ce_scores.parquet"
out_dir = "output"
show_plots = True
save_plots = False

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)

# Load data
cito_df = pd.read_parquet(cito_path)
ce_df = pd.read_parquet(ce_path)

# Pivot data for cross-dataset merges
cito_wide = cito_df.pivot(index=['RINPERSOON', 'year'],
                          columns='assessment', values='value').reset_index()
ce_wide = ce_df.pivot(index=['RINPERSOON', 'year'],
                       columns='assessment', values='value').reset_index()

# 1. Histograms for each assessment
for df, label in [(cito_df, "CITO"), (ce_df, "CE")]:
    for assess in df['assessment'].unique():
        data = df[df['assessment'] == assess]['value'].dropna()
        # Save CSV
        csv_path = os.path.join(out_dir, f"hist_{label}_{assess}.csv")
        data.to_csv(csv_path, index=False)
        # Plot
        plt.figure()
        plt.hist(data, bins=20)
        plt.title(f"{label} {assess} Distribution")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        if save_plots:
            plt.savefig(os.path.join(out_dir, f"hist_{label}_{assess}.png"))
        if show_plots:
            plt.show()
        else:
            plt.close()

# 2. Yearly mean trends
for df, label in [(cito_df, "CITO"), (ce_df, "CE")]:
    trend = df.groupby(['year', 'assessment'])['value'].mean().reset_index()
    csv_path = os.path.join(out_dir, f"yearly_mean_{label}.csv")
    trend.to_csv(csv_path, index=False)
    plt.figure()
    for assess in trend['assessment'].unique():
        sub = trend[trend['assessment'] == assess]
        plt.plot(sub['year'], sub['value'], label=assess)
    plt.title(f"Yearly Mean {label} Scores")
    plt.xlabel("Year")
    plt.ylabel("Mean Score")
    plt.legend()
    if save_plots:
        plt.savefig(os.path.join(out_dir, f"yearly_mean_{label}.png"))
    if show_plots:
        plt.show()
    else:
        plt.close()

# 3. Scatterplots between comparable subjects
# Merge on person and year
merged = pd.merge(cito_wide, ce_wide, on=['RINPERSOON', 'year'], how='inner')
subjects = [
    ("CITOPERCENTIELTAAL", "NLCE"),
    ("CITOPERCENTIELREKENENWISKUNDE", "WISKCE"),
    ("CITOPERCENTIELTAAL", "ENGCE")
]
for cito_sub, ce_sub in subjects:
    if cito_sub in merged.columns and ce_sub in merged.columns:
        subdf = merged[[cito_sub, ce_sub]].dropna()
        csv_path = os.path.join(out_dir, f"scatter_{cito_sub}_vs_{ce_sub}.csv")
        subdf.to_csv(csv_path, index=False)
        plt.figure()
        plt.scatter(subdf[cito_sub], subdf[ce_sub], alpha=0.5)
        plt.title(f"{cito_sub} vs. {ce_sub}")
        plt.xlabel(cito_sub)
        plt.ylabel(ce_sub)
        if save_plots:
            plt.savefig(os.path.join(out_dir, f"scatter_{cito_sub}_vs_{ce_sub}.png"))
        if show_plots:
            plt.show()
        else:
            plt.close()

# 4. Correlation heatmap
all_scores = pd.concat([cito_df, ce_df])
pivot = all_scores.pivot_table(index='RINPERSOON', columns='assessment', values='value')
corr = pivot.corr()
csv_path = os.path.join(out_dir, "correlation_matrix.csv")
corr.to_csv(csv_path)

plt.figure()
plt.imshow(corr, interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Matrix of All Assessments")
plt.tight_layout()
if save_plots:
    plt.savefig(os.path.join(out_dir, "correlation_matrix.png"))
if show_plots:
    plt.show()
else:
    plt.close()

# 5. Individual score changes (delta CITO vs CE Math)
if "CITOPERCENTIELREKENENWISKUNDE" in merged.columns and "WISKCE" in merged.columns:
    merged['delta_math'] = merged['CITOPERCENTIELREKENENWISKUNDE'] - merged['WISKCE']
    delta = merged[['delta_math']].dropna()
    csv_path = os.path.join(out_dir, "delta_math_distribution.csv")
    delta.to_csv(csv_path, index=False)
    plt.figure()
    plt.hist(delta['delta_math'], bins=20)
    plt.title("Distribution of CITO Math Percentile - CE Math Score")
    plt.xlabel("Delta")
    plt.ylabel("Frequency")
    if save_plots:
        plt.savefig(os.path.join(out_dir, "delta_math_distribution.png"))
    if show_plots:
        plt.show()
    else:
        plt.close()

