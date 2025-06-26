import os
import json
import time
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import faiss


def load_embeddings(h5_path):
    with h5py.File(h5_path, "r") as f:
        embeddings = f["embeddings"][:]
    return embeddings


def benchmark_faiss(embeddings, sizes, output_dir):
    dim = embeddings.shape[1]
    results = []

    for n in sizes:
        if n > embeddings.shape[0]:
            continue

        emb_subset = embeddings[:n].astype("float32")

        start = time.time()

        index = faiss.IndexFlatL2(dim)

        
        index.add(emb_subset)
        end = time.time()
        index_time = (end - start) * 1000  # Convert to milliseconds

        print(f"FAISS indexed {n} embeddings in {index_time:.2f} ms")
        results.append({"embeddings": n, "index_time_ms": index_time})

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "faiss_benchmark.csv")
    df.to_csv(csv_path, index=False)

    # Save Plot
    plt.figure()
    plt.plot(df["embeddings"], df["index_time_ms"], marker='o', label="FAISS")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Embeddings")
    plt.ylabel("Index Time (ms)")
    plt.title("FAISS Index Time vs Number of Embeddings")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "faiss_benchmark.png"))

    print(f"Saved results to: {csv_path} and faiss_benchmark.png")


def main(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    emb_path = config["EMBEDDING_PATH"]
    out_dir = config["OUTPUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    emb = load_embeddings(emb_path)
    total = emb.shape[0]
    sizes = [10, 100, 1000, 10000, min(50000, total), min(100000, total), min(200000, total), total]

    benchmark_faiss(emb, sizes, out_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()
    main(args.config)
