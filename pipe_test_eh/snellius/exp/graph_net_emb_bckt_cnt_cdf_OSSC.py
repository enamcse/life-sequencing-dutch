import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', required=True, help='Path to CSV file')
    parser.add_argument('--year', required=True, help='Year for title and filenames')
    args = parser.parse_args()

    csv_file = args.csv_file
    year = args.year

    print(f"✅ Loading CSV: {csv_file}")
    df = pd.read_csv(csv_file)

    # Check required columns
    expected_cols = ['BucketID', 'PopCount', 'LISSCount', 'PopPct', 'LISSPct', 'DiffPct', 'AbsDiffPct']
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    # Sort by PopPct descending
    df_sorted = df.sort_values(by='PopPct', ascending=False).reset_index(drop=True)
    x_values = df_sorted.index + 1

    print("✅ Generating Population Distribution plot...")

    plt.figure(figsize=(12,6))
    plt.bar(x_values, df_sorted['PopPct'])
    plt.xticks(ticks=range(0, len(x_values)+1, 10))
    plt.xlabel('Buckets (sorted by Populations Percentage)')
    plt.ylabel('Population Percentage')
    plt.title(f'Population Distribution Among Buckets - Year {year}')
    plt.tight_layout()

    output_name = f'pop_distribution_{year}.png'
    plt.savefig(output_name)
    print(f"✅ Saved plot: {output_name}")
    plt.close()

    # Compute and plot CDF
    df_sorted['PopPct_cumsum'] = df_sorted['PopPct'].cumsum()

    print("✅ Generating CDF plot...")

    plt.figure(figsize=(12,6))
    plt.plot(x_values, df_sorted['PopPct_cumsum'], marker='o')
    plt.xticks(ticks=range(0, len(x_values)+1, 10))
    plt.ylim(bottom=0)

    plt.xlabel('Buckets (sorted by Populations Percentage)')
    plt.ylabel('Cumulative Population Percentage')
    plt.title(f'Cumulative Distribution Function (CDF) - Year {year}')
    plt.tight_layout()

    output_cdf_name = f'pop_distribution_cdf_{year}.png'
    plt.savefig(output_cdf_name)
    print(f"✅ Saved CDF plot: {output_cdf_name}")
    plt.close()

    print("✅ All plots generated successfully.")

if __name__ == "__main__":
    main()
