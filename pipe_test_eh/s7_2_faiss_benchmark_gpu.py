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

        # Build GPU index
        cpu_index = faiss.IndexFlatL2(dim)
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

        # GPU-based indexing
        gpu_index.add(emb_subset)
        end = time.time()

        indexing_time = (end - start) * 1000  # Convert to milliseconds
        print(f"FAISS GPU indexed {n} embeddings in {indexing_time:.2f} ms")
        results.append({"embeddings": n, "total_time_ms": indexing_time})

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "faiss_gpu_benchmark.csv")
    df.to_csv(csv_path, index=False)

    # Save Plot
    plt.figure()
    plt.plot(df["embeddings"], df["total_time_ms"], marker='o', label="FAISS-GPU")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Embeddings")
    plt.ylabel("Total Time (ms)")
    plt.title("FAISS-GPU Indexing Time vs Number of Embeddings")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "faiss_gpu_benchmark.png"))

    print(f"Saved results to: {csv_path} and faiss_gpu_benchmark.png")



def main(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    emb_path = config["EMBEDDING_PATH"]
    out_dir = config["OUTPUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    emb = load_embeddings(emb_path)
    total = emb.shape[0]
    sizes = []
    val = 10
    while val <= total:
        sizes.append(val)
        val *= 10
    sizes.append(total)

    benchmark_faiss(emb, sizes, out_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()
    main(args.config)
