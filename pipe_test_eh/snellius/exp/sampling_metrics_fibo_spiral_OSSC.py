# name = fibo_metrics_for_ossc.py
# background = this file is created for ossc but it is more debugged on RA machine jupyter notebook
import os, sys
import logging
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import random
from itertools import count
from math import cos, gamma, pi, sin, sqrt
from typing import Callable, Iterator, List, Tuple
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from collections import Counter
from itertools import count
import itertools
import json
from scipy.stats import pearsonr, ks_2samp
from scipy.spatial.distance import jensenshannon

EMBEDDING_FILE = 'EMBEDDING_FILE'
BACKGROUND_FILE = 'BACKGROUND_FILE'
LISS_FILE = 'LISS_FILE'
OUTPUT_DIR = 'OUTPUT_DIR'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fibo_spiral.log')
    ]
)

logger = logging.getLogger(__name__)

# ----------------------------
# 1. Data loading
# ----------------------------
def load_data(embedding_file, background_file):
    logger.info("✅ Loading embedding data...")
    df_emb = pd.read_parquet(embedding_file)
    dim = len(df_emb.columns) - 1  # Exclude rinpersoon_id
    logger.info(f" - Dimension: {dim}")
    logger.info(f" - Rows: {len(df_emb)}, Columns: {list(df_emb.columns)}")

    logger.info("✅ Loading background data...")
    df_bg = pd.read_csv(background_file) if background_file.split('.')[-1] == 'csv'  else pd.read_parquet(background_file)
    logger.info(f" - Rows: {len(df_bg)}, Columns: {list(df_bg.columns)}")

    # Ensure matching ID column names
    if df_emb.columns.isin(["rinpersoon_id"]).any():
        df_emb = df_emb.rename(columns={"rinpersoon_id": "RINPERSOON"})
    if df_bg.columns.isin(["rinpersoon_id"]).any():
        df_bg = df_bg.rename(columns={"rinpersoon_id": "RINPERSOON"})

    logger.info("✅ Joining on RINPERSOON...")
    df_merged = df_emb.merge(df_bg, on="RINPERSOON", how="inner")
    logger.info(f" - Joined rows: {len(df_merged)}")

    logger.info(f"Embedding data # {len(df_emb)} rows, Unique IDs: {df_emb['RINPERSOON'].nunique()}")
    logger.info(f"Background data # {len(df_bg)} rows, Unique IDs: {df_bg['RINPERSOON'].nunique()}")
    return dim, df_merged

# ----------------------------
# 2. Sphere generation
# ----------------------------
def int_sin_m(x: float, m: int) -> float:
    """Computes the integral of sin^m(t) dt from 0 to x recursively"""
    if m == 0:
        return x
    elif m == 1:
        return 1 - cos(x)
    else:
        return (m - 1) / m * int_sin_m(x, m - 2) - cos(x) * sin(x) ** (m - 1) / m

def primes() -> Iterator[int]:
    """Returns an infinite generator of prime numbers"""
    yield from (2, 3, 5, 7)
    composites = {}
    ps = primes()
    next(ps)
    p = next(ps)
    assert p == 3
    psq = p * p
    for i in count(9, 2):
        if i in composites:  # composite
            step = composites.pop(i)
        elif i < psq:  # prime
            yield i
            continue
        else:  # composite, = p*p
            assert i == psq
            step = 2 * p
            p = next(ps)
            psq = p * p
        i += step
        while i in composites:
            i += step
        composites[i] = step

def inverse_increasing(
        func: Callable[[float], float],
        target: float,
        lower: float,
        upper: float,
        atol: float = 1e-10, ) -> float:
    """Returns func inverse of target between lower and upper
    inverse is accurate to an absolute tolerance of atol, and
    must be monotonically increasing over the interval lower
    to upper
    """
    mid = (lower + upper) / 2
    approx = func(mid)
    while abs(approx - target) > atol:
        if approx > target:
            upper = mid
        else:
            lower = mid
        mid = (upper + lower) / 2
        approx = func(mid)
    return mid

def uniform_sphere(d: int, n: int) -> List[List[float]]:
    """Generate n points over the d dimensional hypersphere"""
    assert d > 1
    assert n > 0
    points = [[1 for _ in range(d)] for _ in range(n)]
    for i in range(n):
        t = 2 * pi * i / n
        points[i][0] *= sin(t)
        points[i][1] *= cos(t)
    for dim, prime in zip(range(2, d), primes()):
        offset = sqrt(prime)
        mult = gamma(dim / 2 + 0.5) / gamma(dim / 2) / sqrt(pi)

        def dim_func(y):
            return mult * int_sin_m(y, dim - 1)

        for i in range(n):
            deg = inverse_increasing(dim_func, i * offset % 1, 0, pi)
            for j in range(dim):
                points[i][j] *= sin(deg)
            points[i][dim] *= cos(deg)
    return points

# ----------------------------
# 3. Cone / Bucket assignment
# ----------------------------
def assign_buckets(embeddings: np.ndarray,
                   sphere_points: np.ndarray) -> List[int]:
    """Assign each embedding to the sphere point with max dot product.
    
    Args:
      embeddings: shape (n, d).
      sphere_points: shape (b, d).
    Returns:
      A list of bucket indices of length n.
    """
    dot_prods = np.dot(embeddings, sphere_points.T)
    bucket_ids = np.argmax(dot_prods, axis=1)
    return bucket_ids.tolist()
    

# ----------------------------
# 4. Sampling evenly from cones
# ----------------------------

# Currently, this function is not in use.
def bucket_sampling(words: List[str],
                    bucket_ids: List[int],
                    k: int = 100) -> Tuple[List[str], List[str]]:
    """Sample k words proportionally from each bucket, handling rounding.
    
    Args:
      words: Original words in order.
      bucket_ids: Each word's assigned bucket ID.
      k: Desired sample size.
    Returns:
      (sampled_words, sampled_pos).
    """
    n = len(words)
    bucket_count = Counter(bucket_ids)
    b = len(set(bucket_ids))
    samples_per_bucket = {}
    for bid in bucket_count:
        fraction = bucket_count[bid] / n
        samples_per_bucket[bid] = round(k * fraction)

    diff = k - sum(samples_per_bucket.values())
    if diff != 0:
        # Sort buckets by largest frequency
        sorted_bids = sorted(bucket_count.keys(),
                             key=lambda x: bucket_count[x],
                             reverse=True)
        idx = 0
        while diff != 0 and idx < b:
            if diff > 0:
                samples_per_bucket[sorted_bids[idx]] += 1
                diff -= 1
            else:
                if samples_per_bucket[sorted_bids[idx]] > 0:
                    samples_per_bucket[sorted_bids[idx]] -= 1
                    diff += 1
            idx = (idx + 1) % b

    bucketed_words = {}
    for i, bid in enumerate(bucket_ids):
        if bid not in bucketed_words:
            bucketed_words[bid] = []
        bucketed_words[bid].append(words[i])

    sampled_words = []
    for bid in bucketed_words:
        cnt = samples_per_bucket[bid]
        if cnt > 0:
            indices = list(range(len(bucketed_words[bid])))
            random.shuffle(indices)
            chosen = indices[:cnt]
            for idx in chosen:
                sampled_words.append(bucketed_words[bid][idx])

    return sampled_words


# ----------------------------
# 5. Helper functions for metrics
# ----------------------------
def compute_bucket_percentages(buckets_df, num_buckets=100):
    """Compute the percentage of each bucket in the DataFrame."""
    # bucket_counts = buckets_df.value_counts(normalize=True).sort_index()
    # return bucket_counts.tolist()
    counts = [0] * num_buckets
    for bid in buckets_df:
        counts[bid] += 1
    return [c / len(buckets_df) for c in counts]

def compute_pearson(p, q):
    return pearsonr(p, q)[0]

def compute_ks(p, q):
    return ks_2samp(p, q).statistic

def compute_js(p, q):
    return jensenshannon(p, q)



# ----------------------------
# 6. Compare metadata distribution: Currently not in use
# ----------------------------
def compare_metadata_distribution(full_df, sample_df, variable):
    logger.info(f"\n📊 Comparing distribution for: {variable}")

    full_counts = full_df[variable].value_counts(normalize=True).sort_index()
    sample_counts = sample_df[variable].value_counts(normalize=True).sort_index()

    comparison = pd.DataFrame({
        "Full": full_counts,
        "Sample": sample_counts
    }).fillna(0)
    comparison["AbsDiff"] = (comparison["Full"] - comparison["Sample"]).abs()

    logger.info(comparison)
    return comparison

def save_comparison_to_csv(comparison_df, variable, output_file):
    comparison_df.index.name = variable
    comparison_df.reset_index().to_csv(output_file, index=False)
    logger.info(f"✅ Comparison table saved to {output_file}")

def plot_comparison_distribution(comparison_df, variable, output_file):
    comparison_df = comparison_df.reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.bar(comparison_df[variable] - 0.2, comparison_df["Full"], width=0.4, label="Full")
    plt.bar(comparison_df[variable] + 0.2, comparison_df["Sample"], width=0.4, label="Sample")
    plt.xlabel(variable.capitalize())
    plt.ylabel("Proportion")
    plt.title(f"Distribution Comparison: {variable}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    logger.info(f"✅ Comparison plot saved to {output_file}")

# Helper
def read_json(path):
  with open(path, 'r') as file:
    data = json.load(file)
  return data 

# ----------------------------
# 6. MAIN
# ----------------------------
def main():
    CFG_PATH = sys.argv[1]
    cfg = read_json(CFG_PATH)
    # -------- File paths --------
    
    embedding_file = cfg[EMBEDDING_FILE]
    background_file = cfg[BACKGROUND_FILE]
    liss_file = cfg[LISS_FILE]

    output_dir = cfg[OUTPUT_DIR]
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # -------- Config --------
    dim = 280 # embedding dimension default
    b = 100   # number of sphere directions / buckets default - using a list of num_buckets
    k = 1000  # sample size default - not in use now
    variables_to_compare = ["gender", "year", "month", "municipality"]
    random.seed(42)
    np.random.seed(42)

    embeddings_df = pd.read_csv(embedding_file)
    num_buckets_list = [10, 100, 200] # 500, 1000, 2000 needs really huge time.
    samples_list = ['liss-people']  # sample sets

    param_grid = list(itertools.product(
        embeddings_df.itertuples(index=False),
        num_buckets_list,
        samples_list
    ))

    results = []

    for (embedding_row, num_buckets, sample) in param_grid:
        # Load embeddings from CSV info
        emb_type = embedding_row.embedding_name
        year = embedding_row.year
        file_path = embedding_row.file_path

        logger.info(f"Running: {emb_type}, {year}, {num_buckets}, {sample}")

        # Load embeddings and background data for whole population
        dim, pop_embeddings = load_data(file_path, background_file)
        logger.info('Population Embeddings:', pop_embeddings.head())
        emb_cols = [c for c in pop_embeddings.columns if c.startswith("emb")]

        if len(emb_cols) != dim:
            logger.warning((f"Expected {dim} embedding columns, found {len(emb_cols)}"))
            raise ValueError(f"Expected {dim} embedding columns, found {len(emb_cols)}")

        df_liss = pd.read_parquet(liss_file)
        logger.info(df_liss.head())
        liss_ids = df_liss['RINPERSOON'].unique()
        liss_ids_set = set(liss_ids)

        liss_embeddings = pop_embeddings[pop_embeddings['RINPERSOON'].isin(liss_ids_set)]

        # -------- Generate b=100 points on the d-dim sphere --------
        sphere_pts = uniform_sphere(dim, num_buckets)
        sphere_pts = np.array(sphere_pts)
        logger.info(f"✅ Sphere points generated: {sphere_pts.shape}")

        # Existing stratification and bucketing logic
        pop_buckets = assign_buckets(pop_embeddings[emb_cols], sphere_pts)  # stratify_into_buckets(pop_embeddings, num_buckets)
        liss_buckets = assign_buckets(liss_embeddings[emb_cols], sphere_pts) # stratify_into_buckets(liss_embeddings, num_buckets)

        pop_bucket_pct = compute_bucket_percentages(pop_buckets, num_buckets)
        liss_bucket_pct = compute_bucket_percentages(liss_buckets, num_buckets)

        # Compute metrics
        pearson_corr = compute_pearson(pop_bucket_pct, liss_bucket_pct)
        ks_div = compute_ks(pop_bucket_pct, liss_bucket_pct)
        js_div = compute_js(pop_bucket_pct, liss_bucket_pct)

        max_pct_pop = max(pop_bucket_pct)
        max_pct_liss = max(liss_bucket_pct)

        buckets_covered_pop = sum(p > 0 for p in pop_bucket_pct)
        buckets_covered_liss = sum(p > 0 for p in liss_bucket_pct)

        # Save to results
        results.append({
            'embedding_type': emb_type,
            'year': year,
            'num_buckets': num_buckets,
            'samples': sample,
            'pearson_corr': pearson_corr,
            'ks_div': ks_div,
            'js_div': js_div,
            'max_pct_pop': max_pct_pop,
            'max_pct_liss': max_pct_liss,
            'buckets_covered_pop': buckets_covered_pop,
            'buckets_covered_liss': buckets_covered_liss
        })

    df = pd.DataFrame(results)
    # df.to_csv('sampling_metrics_summary.csv', index=False)
    df.to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)
    logger.info("✅ Sampling metrics saved to 'metrics_summary.csv'")

'''
    # -------- Load & join --------

    df = load_data(embedding_file, background_file)
    

    # -------- Extract embeddings --------
    emb_cols = [c for c in df.columns if c.startswith("emb")]
    embeddings = df[emb_cols].to_numpy()
    person_ids = df["rinpersoon_id"].tolist()

    # -------- Generate b=100 points on the d-dim sphere --------
    sphere_pts = uniform_sphere(dim, b)
    sphere_pts = np.array(sphere_pts)
    print(f"✅ Sphere points generated: {sphere_pts.shape}")

    # -------- Assign people to buckets via cosine sim --------
    bucket_ids = assign_buckets(embeddings, sphere_pts)
    df["bucket"] = bucket_ids
    # Bucket count
    bucket_count = Counter(bucket_ids)
    print(f"✅ Buckets assigned: {len(bucket_count)} unique buckets")
    print(f"Bucket distribution: {bucket_count}")
    print("✅ Buckets assigned")

    # -------- Sample k=1000 people proportionally from buckets --------
    sampled_ids = bucket_sampling(person_ids, bucket_ids, k)
    sampled_df = df[df["rinpersoon_id"].isin(sampled_ids)]
    print("✅ People sampled")

    # -------- Compare distributions --------
    for variable in variables_to_compare:
        print(f"\n✅ Comparing distribution for: {variable}")
        comparison_df = compare_metadata_distribution(df, sampled_df, variable)

        # Save CSV
        csv_file = os.path.join(output_dir, f"comparison_{variable}.csv")
        save_comparison_to_csv(comparison_df, variable, csv_file)

        # Save Plot
        plot_file = os.path.join(output_dir, f"comparison_{variable}.png")
        plot_comparison_distribution(comparison_df, variable, plot_file)
'''

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    main()