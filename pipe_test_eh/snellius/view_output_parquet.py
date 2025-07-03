import pandas as pd
from pathlib import Path
import json

# Path to your Parquet file
seq_file = Path("/projects/0/prjs1019/data/fake_data_v0/step2/background.parquet")

try:
    df = pd.read_parquet(seq_file)
    print(f"\n✅ Loaded {seq_file.name}")
    print(f"Total rows: {len(df)}")
    print("\n📋 Columns found:")
    for col in df.columns:
        print(f"  - {col}")

    print("\n🔍 Showing up to 3 sample rows:\n")
    for i in range(min(3, len(df))):
        print(f"--- Row {i+1} ---")
        row = df.iloc[i]
        for col in df.columns:
            val = row[col]
            print(f"{col}:")
            if isinstance(val, (dict, list)):
                print(json.dumps(val, indent=2))
            else:
                print(val)
            print()
        print("-" * 40)

except Exception as e:
    print(f"[!] Error reading {seq_file}: {e}")

