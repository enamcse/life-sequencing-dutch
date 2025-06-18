import h5py
import numpy as np
from pathlib import Path

# Path to the HDF5 file
h5_path = Path("pipe_test_eh/step5/encoding=mlm/masking=random/encoded.h5")

print(f"🔍 Inspecting HDF5 file: {h5_path}")

with h5py.File(h5_path, "r") as h5_file:
    print("\n📁 Available Datasets:\n")
    for key in h5_file.keys():
        data = h5_file[key]
        print(f"- {key}: shape = {data.shape}, dtype = {data.dtype}")

    print("\n🔍 Sample values from each dataset (first 1–2 entries):\n")
    for key in h5_file.keys():
        data = h5_file[key]
        sample = data[:2]  # Show first 2 samples
        print(f"▶️ {key}:\n{sample}\n")
