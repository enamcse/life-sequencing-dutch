import os
import time
import json
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from annoy import AnnoyIndex


def load_embeddings(h5_path):
    with h5py.File(h5_path, "r") as f:
        embeddings = f["embeddings"][:]
        ids = f["ids"][:]
    return embeddings, ids


def benchmark_annoy(embeddings, sizes, output_dir, trees=10):
    dim = embeddings.shape[1]
    total = embeddings.shape[0]
    results = []

    for n in sizes:
        if n > total:
            continue
        emb_subset = embeddings[:n]

        start = time.time()
        index = AnnoyIndex(dim, 'angular')

        for i, vec in enumerate(emb_subset):
            index.add_item(i, vec.tolist())

        
        index.build(trees)
        index_time = (time.time() - start) * 1000  # Convert to milliseconds

        print(f"Built index for {n} embeddings in {index_time:.2f} ms")
        results.append({"embeddings": n, "index_time_ms": index_time})

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "annoy_benchmark.csv")
    df.to_csv(csv_path, index=False)

    # Save Plot
    plt.figure()
    plt.plot(df["embeddings"], df["index_time_ms"], marker='o')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Embeddings")
    plt.ylabel("Annoy Index Time (ms)")
    plt.title("Annoy Index Time vs Number of Embeddings")
    plt.grid(True)

    plot_path = os.path.join(output_dir, "annoy_benchmark.png")
    plt.savefig(plot_path)
    print(f"Saved output to: {csv_path} and {plot_path}")


def main(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    emb_path = config["EMBEDDING_PATH"]
    out_dir = config["OUTPUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    emb, ids = load_embeddings(emb_path)
    total = emb.shape[0]

    sizes = []
    val = 10
    while val <= total:
        sizes.append(val)
        val *= 10
    sizes.append(total)
    benchmark_annoy(emb, sizes, out_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()
    main(args.config)
