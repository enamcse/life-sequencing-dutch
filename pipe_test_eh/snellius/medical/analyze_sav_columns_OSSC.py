import argparse
import os
import pandas as pd
import pyreadstat
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory containing .sav files')
    parser.add_argument('--output_dir', default='.', help='Directory to save output files')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"✅ Scanning directory: {input_dir}")

    # Find .sav files
    sav_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.sav')]
    if not sav_files:
        print("❌ No .sav files found in the directory!")
        return

    print(f"✅ Found {len(sav_files)} .sav files")

    # Map of file -> set of columns
    file_columns = {}
    all_columns = set()
    column_file_map = defaultdict(set)

    for filepath in sav_files:
        try:
            print(f"▶️ Reading: {filepath}")
            df, meta = pyreadstat.read_sav(filepath)
            columns = set(df.columns)
            file_columns[filepath] = columns
            all_columns.update(columns)
            for col in columns:
                column_file_map[col].add(os.path.basename(filepath))
        except Exception as e:
            print(f"❌ Error reading {filepath}: {e}")

    if not file_columns:
        print("❌ No valid .sav files were processed!")
        return

    # Find common columns (intersection)
    common_columns = set.intersection(*file_columns.values())

    print(f"✅ Found {len(common_columns)} columns common to ALL files.")

    # Count occurrence of all other columns
    extra_columns = all_columns - common_columns
    extra_col_counts = {col: len(column_file_map[col]) for col in extra_columns}
    sorted_extra_cols = sorted(extra_col_counts.items(), key=lambda x: -x[1])

    print(f"✅ Found {len(extra_columns)} extra columns with varying presence.")

    # Save common columns
    common_outfile = os.path.join(output_dir, "common_columns.txt")
    with open(common_outfile, 'w') as f:
        for col in sorted(common_columns):
            f.write(col + '\n')
    print(f"✅ Saved common columns list to {common_outfile}")

    # Save extra columns summary
    extra_summary = []
    for col, count in sorted_extra_cols:
        files = sorted(column_file_map[col])
        extra_summary.append({
            "Column": col,
            "Occurrences": count,
            "Files": "; ".join(files)
        })

    extra_outfile = os.path.join(output_dir, "extra_columns_summary.csv")
    pd.DataFrame(extra_summary).to_csv(extra_outfile, index=False)
    print(f"✅ Saved extra columns summary to {extra_outfile}")

    print("🎯 Analysis complete.")

if __name__ == "__main__":
    main()
