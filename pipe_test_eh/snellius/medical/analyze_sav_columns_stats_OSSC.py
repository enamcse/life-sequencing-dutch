import argparse
import os
import pandas as pd
import numpy as np
import pyreadstat
from collections import defaultdict

def is_numeric_dtype(series):
    return pd.api.types.is_numeric_dtype(series)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory containing .sav files')
    parser.add_argument('--output_dir', required=True, help='Directory to save output files')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"✅ Scanning directory: {input_dir}")

    sav_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.sav')]
    if not sav_files:
        print("❌ No .sav files found!")
        return

    print(f"✅ Found {len(sav_files)} .sav files")

    # Collect all columns present in all files
    all_columns_set = set()
    file_columns_map = defaultdict(set)

    # Store per-column aggregated stats
    col_file_count = defaultdict(int)
    col_datatypes = {}
    col_sums = defaultdict(float)
    col_mins = defaultdict(list)
    col_maxs = defaultdict(list)
    col_avgs = defaultdict(list)
    col_medians = defaultdict(list)
    col_stds = defaultdict(list)
    col_zeros = defaultdict(int)
    col_nulls = defaultdict(int)
    col_total_rows = defaultdict(int)
    col_file_sums = defaultdict(list)  # For annual average sum

    for filepath in sav_files:
        try:
            print(f"▶️ Reading: {filepath}")
            df, meta = pyreadstat.read_sav(filepath)
            n_rows = len(df)

            for col in df.columns:
                series = df[col]
                all_columns_set.add(col)
                file_columns_map[col].add(os.path.basename(filepath))
                col_file_count[col] += 1
                col_total_rows[col] += n_rows

                non_null_series = series.dropna()
                empty_count = n_rows - non_null_series.shape[0]
                col_nulls[col] += empty_count

                if is_numeric_dtype(series):
                    col_datatypes[col] = "Numeric"

                    zero_count = (non_null_series == 0).sum()
                    col_zeros[col] += zero_count

                    # Basic stats
                    total_sum = non_null_series.sum()
                    col_sums[col] += total_sum
                    col_file_sums[col].append(total_sum)

                    if not non_null_series.empty:
                        col_mins[col].append(non_null_series.min())
                        col_maxs[col].append(non_null_series.max())
                        col_avgs[col].append(non_null_series.mean())
                        col_medians[col].append(non_null_series.median())
                        col_stds[col].append(non_null_series.std())
                else:
                    col_datatypes[col] = "String"

        except Exception as e:
            print(f"❌ Error reading {filepath}: {e}")

    print("✅ Aggregation complete across all files.")

    # Prepare final summary
    summary_rows = []
    for col in sorted(all_columns_set):
        file_count = col_file_count.get(col, 0)
        data_type = col_datatypes.get(col, "Unknown")
        total_rows = col_total_rows.get(col, 0)
        empty_rows = col_nulls.get(col, 0)
        percent_exists = (1 - empty_rows / total_rows) * 100 if total_rows else 0

        if data_type == "Numeric":
            sum_value = col_sums.get(col, 0)
            max_value = max(col_maxs[col]) if col_maxs[col] else None
            min_value = min(col_mins[col]) if col_mins[col] else None
            avg_value = np.mean(col_avgs[col]) if col_avgs[col] else None
            median_value = np.median(col_medians[col]) if col_medians[col] else None
            std_dev = np.mean(col_stds[col]) if col_stds[col] else None
            zero_rows = col_zeros.get(col, 0)
            annual_avg_sum = np.mean(col_file_sums[col]) if col_file_sums[col] else None
        else:
            sum_value = max_value = min_value = avg_value = median_value = std_dev = zero_rows = annual_avg_sum = None

        summary_rows.append({
            "Column_name": col,
            "File Count": file_count,
            "DataType": data_type,
            "Sum": sum_value,
            "Max": max_value,
            "Avg": avg_value,
            "Median": median_value,
            "Min": min_value,
            "Std Dev": std_dev,
            "Empty Rows": empty_rows,
            "Zero Rows": zero_rows,
            "Total Rows": total_rows,
            "Percent Value Exists": percent_exists,
            "Annual average sum": annual_avg_sum
        })

    summary_df = pd.DataFrame(summary_rows)
    output_file = os.path.join(output_dir, "column_statistics_summary.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"✅ Summary saved to {output_file}")

    print("🎯 Done.")

if __name__ == "__main__":
    main()
