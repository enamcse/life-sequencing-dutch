import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with parquet files")
    parser.add_argument("--prefix", required=True, help="Prefix to filter parquet files")
    parser.add_argument("--output_file", required=True, help="Output merged parquet file path")
    args = parser.parse_args()

    input_dir = args.input_dir
    prefix = args.prefix
    output_file = args.output_file

    print(f"✅ Scanning directory: {input_dir}")
    print(f"✅ Using prefix: {prefix}")

    # Find parquet files with prefix
    parquet_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.startswith(prefix) and f.lower().endswith(".parquet")
    ]

    if not parquet_files:
        print("❌ No matching parquet files found!")
        return

    print(f"✅ Found {len(parquet_files)} matching parquet files.")
    for f in parquet_files:
        print(f" - {f}")

    # Load and concatenate
    dataframes = []
    for file in parquet_files:
        print(f"▶️ Reading: {file}")
        df = pd.read_parquet(file)
        dataframes.append(df)

    print("✅ Concatenating DataFrames...")
    merged_df = pd.concat(dataframes, ignore_index=True)

    print(f"✅ Writing merged parquet to: {output_file}")
    merged_df.to_parquet(output_file, index=False)

    print("🎯 Merge complete!")

if __name__ == "__main__":
    main()
