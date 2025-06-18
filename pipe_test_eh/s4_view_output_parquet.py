import pandas as pd
from pathlib import Path
import json

# Path to your step4 output
seq_file = Path("pipe_test_eh/step4/sequences.parquet")

try:
    df = pd.read_parquet(seq_file)
    print(f"\n✅ Loaded {seq_file.name}")
    print(f"Total rows: {len(df)}")
    print("Columns:", df.columns.tolist())

    # Show a few samples
    print("\n🔍 Showing 3 example sequences:\n")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"--- RINPERSOON: {row['RINPERSOON']} ---")
        print("Background:", json.dumps(row['background'], indent=2))
        print("Sentence:")
        for j, event in enumerate(row['sentence']):
            print(f"  {j+1:>2}. {' | '.join(event)}")
        print("Abspos:", row['abspos'])
        print("Age:", row['age'])
        print("Segment:", row['segment'])
        print()

except Exception as e:
    print(f"[!] Error reading {seq_file}: {e}")
