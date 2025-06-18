import pandas as pd
from pathlib import Path

# Set path to your step3 directory
step3_dir = Path("pipe_test_eh/step3")

# List of parquet files to inspect
files = sorted(step3_dir.glob("*.parquet"))

for file in files:
    print(f"\n📁 File: {file.name}")
    try:
        df = pd.read_parquet(file)
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("Sample rows:")
        print(df.head(3).to_string(index=False))
    except Exception as e:
        print(f"[!] Failed to load {file.name}: {e}")
