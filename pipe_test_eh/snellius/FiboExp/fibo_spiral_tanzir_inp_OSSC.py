# name = fibo_metrics_for_ossc.py
# background = this file is created for ossc but it is more debugged on RA machine jupyter notebook
# The following lines should be run in a terminal:
# <array_jobid> should be replaced with the actual job ID from the sbatch command.
# sbatch --array=0-13 sampling_metrics_fibo_spiral_OSSC.sh
# sbatch --dependency=afterok:<array_jobid> sampling_metrics_fibo_spiral_OSSC_post_merge.sh

import argparse
import os
import sys
import traceback
import logging
from types import SimpleNamespace
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
from scipy.stats import pearsonr, ks_2samp, chisquare, wasserstein_distance, entropy, spearmanr
from scipy.spatial.distance import jensenshannon, cosine
import torch
from dataclasses import dataclass
from tqdm import tqdm
from joblib import Parallel, delayed
import psutil
import gc
import csv

EMBEDDING_FILE = 'EMBEDDING_FILE'
BACKGROUND_FILE = 'BACKGROUND_FILE'
LISS_FILE = 'LISS_FILE'
OUTPUT_DIR = 'OUTPUT_DIR'
OUTPUT_POP_BUCKET_ID = 'OUTPUT_POP_BUCKET_ID'
OUTPUT_LISS_BUCKET_ID = 'OUTPUT_LISS_BUCKET_ID'
OUTPUT_BUCKET_SUMMARY = 'OUTPUT_BUCKET_SUMMARY'
OUTPUT_RINPERSOON_YEAR_BUCKET = 'OUTPUT_RINPERSOON_YEAR_BUCKET'
SPHERE_POINTS_DIR = 'SPHERE_POINTS_DIR'
NUM_BUCKETS_LIST = 'NUM_BUCKETS_LIST'
DO_WHITENING = 'DO_WHITENING'
DO_WHITENING_CORR = 'DO_WHITENING_CORR'
MERGE_ALL_EMBEDDING_FILES = 'MERGE_ALL_EMBEDDING_FILES'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fibo_spiral.log')
    ]
)

logger = logging.getLogger(__name__)

def log_memory(tag=""):
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1e9
    logging.info(f" Memory used {tag}: {mem_gb:.2f} GB")

# ----------------------------
# 0. Whitening function
# ----------------------------
def whitening_torch_final(embeddings):
    mu = torch.mean(embeddings, dim=0, keepdim=True)
    cov = torch.mm((embeddings - mu).t(), embeddings - mu)
    u, s, vt = torch.svd(cov)
    W = torch.mm(u, torch.diag(1/torch.sqrt(s)))
    embeddings = torch.mm(embeddings - mu, W)
    return embeddings

def streamed_whitening(df_emb, emb_cols, chunk_size=100_000):
    """
    1) Compute global mean μ in streaming fashion
    2) Compute global covariance Σ = Σ_i (x_i-μ)(x_i-μ)^T
    3) SVD on Σ → U, S
    4) For each chunk, apply (x-μ)·U·diag(1/√S)
    """
    n = len(df_emb)
    dim = len(emb_cols)

    # 1) compute mean
    log_memory("before mean pass")
    total = np.zeros(dim, dtype=np.float64)
    count = 0
    for start in range(0, n, chunk_size):
        chunk = df_emb.iloc[start:start+chunk_size][emb_cols].to_numpy(dtype=np.float64)
        total += chunk.sum(axis=0)
        count += chunk.shape[0]
        del chunk; gc.collect()
    mu = total / count
    log_memory("after mean pass")

    # 2) compute covariance
    cov = np.zeros((dim, dim), dtype=np.float64)
    for start in range(0, n, chunk_size):
        chunk = df_emb.iloc[start:start+chunk_size][emb_cols].to_numpy(dtype=np.float64)
        centered = chunk - mu
        cov += centered.T @ centered
        del chunk, centered; gc.collect()
    cov /= (count - 1)
    log_memory("after covariance pass")

    # 3) SVD + whitening matrix W
    cov_t = torch.from_numpy(cov.astype(np.float32))
    u, s, _ = torch.svd(cov_t)
    W = u @ torch.diag(1.0 / torch.sqrt(s))
    log_memory("after SVD/whitening matrix")

    # 4) apply transformation in chunks and rebuild df_emb
    whitened_cols = []
    for start in range(0, n, chunk_size):
        chunk = df_emb.iloc[start:start+chunk_size][emb_cols].to_numpy(dtype=np.float32)
        centered = torch.from_numpy(chunk) - torch.from_numpy(mu.astype(np.float32))
        whitened = centered @ W
        whitened_cols.append(whitened.numpy())
        del chunk, centered, whitened; gc.collect()

    # stitch back into one array
    all_whitened = np.vstack(whitened_cols)
    log_memory("after assembling whitened array")

    return all_whitened  # shape (n, dim)

def streamed_whitening_df(df_emb, embedding_cols, chunk_size=100_000):
    """
    Chunked whitening directly from a Pandas DataFrame.
    1) streaming mean
    2) streaming covariance
    3) SVD → whitening matrix W
    4) streaming apply W to each chunk
    """
    n = len(df_emb)
    d = len(embedding_cols)

    # 1️⃣ Streaming mean
    log_memory("start mean pass")
    total = np.zeros(d, dtype=np.float64)
    count = 0
    for start in range(0, n, chunk_size):
        chunk = df_emb.iloc[start:start+chunk_size][embedding_cols] \
                    .to_numpy(dtype=np.float64)
        total += chunk.sum(axis=0)
        count += chunk.shape[0]
        del chunk; gc.collect()
    mu = total / count
    log_memory("after mean pass")

    # 2️⃣ Streaming covariance
    cov = np.zeros((d, d), dtype=np.float64)
    for start in range(0, n, chunk_size):
        chunk = df_emb.iloc[start:start+chunk_size][embedding_cols] \
                    .to_numpy(dtype=np.float64)
        centered = chunk - mu
        cov += centered.T @ centered
        del chunk, centered; gc.collect()
    cov /= (count - 1)
    log_memory("after covariance pass")

    # 3️⃣ SVD → compute W
    u, s, vt = np.linalg.svd(cov, full_matrices=False)
    W = u @ np.diag(1.0 / np.sqrt(s))
    log_memory("after SVD / whitening matrix")

    # 4️⃣ Apply whitening in chunks
    whitened = np.empty((n, d), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = df_emb.iloc[start:end][embedding_cols] \
                    .to_numpy(dtype=np.float32)
        centered = chunk - mu.astype(np.float32)
        whitened[start:end] = centered @ W.astype(np.float32)
        del chunk, centered; gc.collect()
    log_memory("after applying whitening")

    return whitened



# ----------------------------
# 0. Evaluation Helper
# ----------------------------
@dataclass
class CorrelationMetrics:
    pearson: float
    spearman: float
    pearson_pval: float
    spearman_pval: float

class Evaluator:
    """Computes correlation metrics on a validation set."""

    def __init__(self, rng: np.random.Generator):
        self._rng = rng

    def pairwise_cosines(self, vectors: np.ndarray, permuted_vectors: np.ndarray) -> np.ndarray:
        """Vectorised cosine similarity for two aligned matrices."""
        dot = np.sum(vectors * permuted_vectors, axis=1)
        norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(permuted_vectors, axis=1)
        return dot / norms

    def correlation(self, target_vectors: np.ndarray, aligned_vectors: np.ndarray) -> CorrelationMetrics:
        n = target_vectors.shape[0]
        permutation = self._rng.permutation(n)

        # Compute cosine similarities for permuted pairs
        cos_target = self.pairwise_cosines(target_vectors, target_vectors[permutation])
        cos_aligned = self.pairwise_cosines(aligned_vectors, aligned_vectors[permutation])

        # Compute Pearson and Spearman correlations between those similarities
        pearson_corr, pearson_pval = pearsonr(cos_target, cos_aligned, alternative='greater')
        spearman_corr, spearman_pval = spearmanr(cos_target, cos_aligned, alternative='greater')

        return CorrelationMetrics(
            pearson=pearson_corr,
            spearman=spearman_corr,
            pearson_pval=pearson_pval,
            spearman_pval=spearman_pval
        )


# ----------------------------
# 1. Data loading
# ----------------------------
def load_and_merge_embeddings(csv_path):
    df_paths = pd.read_csv(csv_path)
    merged_df = pd.DataFrame()
    for i, row in df_paths.iterrows():
        
        path = row['path'] if 'path' in row else row[2]
        year = row['year'] if 'year' in row else row[1]
        logging.info(f"📂 Loading embedding file: {path}")
        
        log_memory(f"Before loading file: {path}")
        df = pd.read_parquet(path)
        df['year'] = year
        merged_df = pd.concat([merged_df, df], ignore_index=True)
        del df
        gc.collect()
        log_memory(f"After loading file: {path}")
    
    log_memory(f"After concatenating all files")
    logging.info(f"✅ Merged {len(merged_df)} embedding files with shape: {merged_df.shape}")
    return merged_df

def load_data(embedding_file, background_file, cfg, emb_type=None):
    logger.info("✅ Loading embedding data...")
    if cfg.get(MERGE_ALL_EMBEDDING_FILES, 0):
        df_emb = load_and_merge_embeddings(embedding_file)
    else:
        df_emb = pd.read_parquet(embedding_file)

    # Identify embedding columns
    embedding_cols = [col for col in df_emb.columns if col.lower().startswith("emb")]
    meta_cols = [col for col in df_emb.columns if not col.lower().startswith("emb")]

    logger.info(f" - Meta columns: {meta_cols}")
    logger.info(f" - Embedding columns: {embedding_cols}")

    dim = len(embedding_cols)  # Exclude rinpersoon_id, year, etc
    logger.info(f" - Dimension: {dim}")
    logger.info(f" - Rows: {len(df_emb)}, Columns: {list(df_emb.columns)}")

    if cfg.get(DO_WHITENING, 0):
        log_memory("✅ Whitening embeddings as per config")

        # 1) Grab one single NumPy view of all embeddings (no copy)
        emb_np = df_emb[embedding_cols].to_numpy(dtype=np.float32, copy=False)
        log_memory("✅ Grabbed embeddings as NumPy array")

        # Preserve for correlation, if requested
        if cfg.get(DO_WHITENING_CORR, 0):
            emb_before_np = emb_np

        # 2) Streamed whitening in chunks
        log_memory("START whitening")
        whitened_np = streamed_whitening_np(emb_np, chunk_size=100_000)
        log_memory("DONE whitening")

        # 3) Rebuild df_emb from whitened_np + meta columns
        df_emb_whitened = pd.DataFrame(whitened_np, columns=embedding_cols)
        for col in meta_cols:
            df_emb_whitened[col] = df_emb[col].values
        df_emb = df_emb_whitened[meta_cols + embedding_cols]
        logger.info("✅ Whitening complete")

        # 4) Compute correlation if requested
        if cfg.get(DO_WHITENING_CORR, 0):
            logger.info("✅ Evaluating whitening distortion (correlation)")
            emb_after_np = df_emb[embedding_cols].to_numpy(dtype=np.float32, copy=False)

            rng = np.random.default_rng(seed=42)
            evaluator = Evaluator(rng)
            corr_metrics = evaluator.correlation(emb_before_np, emb_after_np)

            logger.info(
                f"✅ Whitening Correlation – "
                f"Pearson: {corr_metrics.pearson:.4f} (p={corr_metrics.pearson_pval:.4f}); "
                f"Spearman: {corr_metrics.spearman:.4f} (p={corr_metrics.spearman_pval:.4f})"
            )

            output_dir = cfg[OUTPUT_DIR]
            os.makedirs(output_dir, exist_ok=True)
            emb_name = emb_type or "default"
            corr_file = os.path.join(output_dir, f"whitening_corr_{emb_name}.csv")
            pd.DataFrame([{
                "embedding_name": emb_name,
                "pearson": corr_metrics.pearson,
                "spearman": corr_metrics.spearman,
                "pearson_pval": corr_metrics.pearson_pval,
                "spearman_pval": corr_metrics.spearman_pval
            }]).to_csv(corr_file, index=False)
            logger.info(f"✅ Whitening correlation metrics saved to {corr_file}")

    else:
        logger.info("⚠️ Whitening skipped (DO_WHITENING=0)")


    log_memory('Before renaming rinpersoon')

    if df_emb.columns.isin(["rinpersoon_id"]).any():
        df_emb.rename(columns={"rinpersoon_id": "RINPERSOON"}, inplace=True)
    logger.info(f"Embedding data # {len(df_emb)} rows, Unique IDs: {df_emb['RINPERSOON'].nunique()}")

    log_memory('After renaming rinpersoon')

    if not cfg.get(BACKGROUND_FILE, 0) is None:
        logger.info("✅ Loading background data...")
        df_bg = pd.read_csv(background_file) if background_file.split('.')[-1] == 'csv'  else pd.read_parquet(background_file)
        logger.info(f" - Rows: {len(df_bg)}, Columns: {list(df_bg.columns)}")

        # Ensure matching ID column names
        if df_bg.columns.isin(["rinpersoon_id"]).any():
            df_bg = df_bg.rename(columns={"rinpersoon_id": "RINPERSOON"})

        logger.info("✅ Joining on RINPERSOON...")
        df_merged = df_emb.merge(df_bg, on="RINPERSOON", how="inner")
        logger.info(f" - Joined rows: {len(df_merged)}")
        
        logger.info(f"Background data # {len(df_bg)} rows, Unique IDs: {df_bg['RINPERSOON'].nunique()}")
    else:
        df_merged = df_emb
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

def load_or_compute_sphere(dim, num_buckets, sphere_dir):
    filename = os.path.join(
        sphere_dir,
        f"sphere_{dim}D_{num_buckets}pts.npy"
    )
    if os.path.exists(filename):
        logger.info(f"✅ Loading cached sphere points from {filename}")
        return np.load(filename)
    logger.info(f"⚠️  Sphere points not found, computing...")
    pts = uniform_sphere(dim, num_buckets)

    os.makedirs(sphere_dir, exist_ok=True)
    np.save(filename, pts)
    logger.info(f"✅ Saved sphere points to {filename}")
    return pts


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

# chunked version for large datasets
def assign_buckets_chunked(embeddings, sphere_points, chunk_size=5_000_000):
    results = []
    for i in tqdm(range(0, len(embeddings), chunk_size), desc="Assigning Buckets"):
        chunk = embeddings[i:i+chunk_size]
        dot_prods = np.dot(chunk, sphere_points.T)
        bucket_ids = np.argmax(dot_prods, axis=1)
        results.extend(bucket_ids.tolist())
    return results

# Parallelized version for large datasets
def process_chunk(chunk, sphere_points):
    dot_prods = np.dot(chunk, sphere_points.T)
    bucket_ids = np.argmax(dot_prods, axis=1)
    del dot_prods
    gc.collect()
    return bucket_ids

def assign_buckets_parallel(embeddings, sphere_points, output_file, chunk_size=5_000_000, n_jobs=4):
    n = len(embeddings)
    log_memory(f"Assign Buckets - Start")

    def generate_chunks():
        for i in range(0, n, chunk_size):
            yield embeddings[i:i+chunk_size]

    with open(output_file, 'w') as f:
        f.write('bucket_id\n')

        with tqdm(total=n, desc="Assigning Buckets") as pbar:
            for bucket_list in Parallel(n_jobs=n_jobs, backend='loky', prefer='processes')(
                delayed(process_chunk)(chunk, sphere_points) for chunk in generate_chunks()
            ):
                for bucket_id in bucket_list:
                    f.write(f"{bucket_id}\n")
                pbar.update(len(bucket_list))
                gc.collect()
                log_memory(f"After chunk processed")

    log_memory(f"Assign Buckets - Done")

    return np.concatenate(results).tolist()

def stratify_population(pop_embeddings, emb_cols, sphere_pts, cfg):
    pop_buckets = []

    if cfg.get(MERGE_ALL_EMBEDDING_FILES, 0):
        logger.info("🔀 MERGE_ALL_EMBEDDING_FILES: streaming bucket assignment in chunks")
        n = len(pop_embeddings)
        chunk_size = 100_000

        # iterate by row‑index chunks
        for start in tqdm(range(0, n, chunk_size), desc="Bucketing pop embeddings"):
            end = min(start + chunk_size, n)
            chunk_df = pop_embeddings.iloc[start:end]
            emb_chunk = chunk_df[emb_cols].to_numpy(dtype=np.float32)

            # compute bucket IDs
            dot_prods = emb_chunk.dot(sphere_pts.T)          # (chunk_size × num_buckets)
            bucket_ids = np.argmax(dot_prods, axis=1)        # length = chunk_size

            # accumulate
            pop_buckets.extend(bucket_ids.tolist())

            # clean up
            del emb_chunk, dot_prods, bucket_ids, chunk_df
            gc.collect()

        log_memory("After streaming bucketing")

    else:
        logger.info("🔀 Single-call bucket assignment")
        embedding_array = pop_embeddings[emb_cols].to_numpy(copy=False, dtype=np.float32)
        log_memory("Before assign_buckets_parallel")
        pop_buckets = assign_buckets_parallel(embedding_array, sphere_pts,
                                              chunk_size=1_000_000, n_jobs=cfg.get("N_JOBS", 4))
        log_memory("After assign_buckets_parallel")

    # now pop_buckets is a flat list of length len(pop_embeddings)
    # and exactly what your SAVE section expects:
    #   'RINPERSOON': pop_embeddings['RINPERSOON'], 'YEAR': year, 'BucketID': pop_buckets

    # return it for later saving
    return pop_buckets


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
def expand_counts_to_samples(counts):
    samples = []
    for idx, count in enumerate(counts):
        samples.extend([idx] * int(count))
    return np.array(samples)

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

def compute_js(p, q):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    p /= p.sum()
    q /= q.sum()
    return jensenshannon(p, q)

def compute_ks_from_samples(p_samples, q_samples):
    return ks_2samp(p_samples, q_samples).statistic

def compute_kl(p, q, epsilon=1e-10):
    p = np.array(p, dtype=float) + epsilon
    q = np.array(q, dtype=float) + epsilon
    p /= p.sum()
    q /= q.sum()
    return np.sum(p * np.log(p / q))

def compute_chi_square(pop_counts, liss_counts):
    pop_counts = np.array(pop_counts, dtype=float)
    liss_counts = np.array(liss_counts, dtype=float)
    expected = pop_counts / pop_counts.sum() * liss_counts.sum()
    expected += 1e-10
    liss_counts += 1e-10
    stat, _ = chisquare(f_obs=liss_counts, f_exp=expected)
    return stat

def compute_wasserstein(p_samples, q_samples):
    return wasserstein_distance(p_samples, q_samples)

def compute_hellinger(p, q):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    p /= p.sum()
    q /= q.sum()
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))

def compute_entropy(probs):
    probs = np.array(probs, dtype=float) + 1e-10
    probs /= probs.sum()
    return entropy(probs)

def compute_gini(counts):
    counts = np.sort(np.array(counts, dtype=float))
    n = len(counts)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * counts) / np.sum(counts) - (n + 1)) / n

def compute_cosine_similarity(p, q):
    return 1 - cosine(p, q)

def compute_coverage(probs, threshold=1e-6):
    return sum(p > threshold for p in probs)

def buckets_to_cover_fraction(probs, target=0.9):
    sorted_probs = sorted(probs, reverse=True)
    cumulative = 0.0
    count = 0
    for p in sorted_probs:
        cumulative += p
        count += 1
        if cumulative >= target:
            return count
    return len(probs)


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

def build_param_grid(cfg):
    embedding_file = cfg[EMBEDDING_FILE]
    embeddings_df = pd.read_csv(embedding_file)
    num_buckets_list = cfg[NUM_BUCKETS_LIST]
    samples_list = ['liss-people']  # sample sets

    return list(itertools.product(
        embeddings_df.itertuples(index=False),
        num_buckets_list,
        samples_list
    ))

# ----------------------------
# 7. Embedding Processing Function
# ----------------------------
def process_embedding(embedding_row, num_buckets, sample, cfg):
    try:
        # ----- Load configuration and data -----
        emb_type = embedding_row.embedding_name
        year = embedding_row.year
        file_path = embedding_row.file_path

        background_file = cfg[BACKGROUND_FILE]
        liss_file = cfg[LISS_FILE]
        output_dir = os.path.expanduser(cfg[OUTPUT_DIR])
        os.makedirs(output_dir, exist_ok=True)
        sphere_dir = cfg.get(SPHERE_POINTS_DIR, output_dir)

        logger.info(f"Running: {emb_type}, {year}, {num_buckets}, {sample}")
        
        log_memory("Before loading data")
        # Load embeddings and background data for whole population
        dim, pop_embeddings = load_data(file_path, background_file, cfg, emb_type)
        logger.info(f'Population Embeddings:\n{pop_embeddings.head()}')
        emb_cols = [c for c in pop_embeddings.columns if c.startswith("emb")]
        log_memory("After loading data")

        if len(emb_cols) != dim:
            logger.warning((f"Expected {dim} embedding columns, found {len(emb_cols)}"))
            raise ValueError(f"Expected {dim} embedding columns, found {len(emb_cols)}")

        if not cfg.get(LISS_FILE, 0) is None:
            # Load LISS dataset
            df_liss = pd.read_parquet(liss_file)
            logger.info(f'LISS Embeddings:\n{df_liss.head()}')
            liss_ids = df_liss['RINPERSOON'].unique()
            liss_ids_set = set(liss_ids)

            liss_embeddings = pop_embeddings[pop_embeddings['RINPERSOON'].isin(liss_ids_set)]
        log_memory("Before Sphere Load")
        # Generate or load sphere points
        sphere_pts = load_or_compute_sphere(dim, num_buckets, sphere_dir)
        sphere_pts = np.array(sphere_pts)
        logger.info(f"✅ Sphere points generated: {sphere_pts.shape}")

        # --------------- STRATIFICATION -----------------
        pop_buckets = stratify_population(
            pop_embeddings,
            emb_cols,
            sphere_pts,
            cfg
        )

        # ---------- SAVE BUCKET IF NEEDED --------------
        if cfg.get(OUTPUT_POP_BUCKET_ID, 0):
            pop_buckets_output = pd.DataFrame({
                'RINPERSOON': pop_embeddings['RINPERSOON'],
                'YEAR': pop_embeddings['year'],
                'BUCKET_ID': pop_buckets
            })
            pop_buckets_output_filename = f"pop_bucket_ids_{emb_type}_buckets{num_buckets}.parquet"
            pop_buckets_output.to_parquet(
                os.path.join(output_dir, pop_buckets_output_filename),
                index=False
            )
            logger.info(f"✅ Population bucket IDs saved to '{pop_buckets_output_filename}'")

        if cfg.get(OUTPUT_LISS_BUCKET_ID, 0) and not cfg.get(LISS_FILE, 0) is None:
            liss_buckets_output = pd.DataFrame({
                'RINPERSOON': liss_embeddings['RINPERSOON'],
                'YEAR':year,
                'BucketID': liss_buckets
            })
            liss_buckets_output_filename = f"liss_bucket_ids_{embedding_file_base}_buckets{num_buckets}.parquet"
            liss_buckets_output.to_parquet(os.path.join(output_dir, liss_buckets_output_filename), index=False)
            logger.info(f"✅ LISS bucket IDs saved to '{liss_buckets_output_filename}'")

        # --------------- COUNTS & PROBABILITIES -----------------
        pop_counts = [0] * num_buckets
        for bid in tqdm(pop_buckets,desc='Counting Population in each bucket'):
            pop_counts[bid] += 1
        if not cfg.get(LISS_FILE, 0) is None:
            liss_counts = [0] * num_buckets
            for bid in liss_buckets:
                liss_counts[bid] += 1

        pop_total = sum(pop_counts)
        if not cfg.get(LISS_FILE, 0) is None:
            liss_total = sum(liss_counts)

        pop_probs = [c / pop_total for c in pop_counts]
        if not cfg.get(LISS_FILE, 0) is None:
            liss_probs = [c / liss_total for c in liss_counts]

        # --------------- SAVE BUCKET SUMMARY -----------------
        if cfg.get(OUTPUT_BUCKET_SUMMARY, 0):
            bucket_df = pd.DataFrame({
                'BucketID': list(range(num_buckets)),
                'PopCount': pop_counts,
                'LISSCount': '' if cfg.get(LISS_FILE, 0) is None else liss_counts,
                'PopPct': pop_probs,
                'LISSPct': '' if cfg.get(LISS_FILE, 0) is None else liss_probs
            })
            if not cfg.get(LISS_FILE, 0) is None:
                bucket_df['DiffPct'] = bucket_df['PopPct'] - bucket_df['LISSPct']
                bucket_df['AbsDiffPct'] = bucket_df['DiffPct'].abs()

            summary_filename = f"bucket_summary_{emb_type}_buckets{num_buckets}.csv"
            bucket_df.to_csv(os.path.join(output_dir, summary_filename), index=False)
            logger.info(f"✅ Bucket summary saved to '{summary_filename}'")

        # --------------- RINPERSOON-YEAR-BUCKET -----------------
        if cfg.get(OUTPUT_RINPERSOON_YEAR_BUCKET, 0) and not cfg.get(MERGE_ALL_EMBEDDING_FILES, 0):
            pop_bucket_df = pd.DataFrame({
                'RINPERSOON': pop_embeddings['RINPERSOON'],
                'YEAR': embedding_row.year,
                'BUCKET_ID': pop_buckets
            })
            pop_filename = f"population_rinpersoon_year_bucket_{emb_type}_buckets{num_buckets}.parquet"
            pop_bucket_df.to_parquet(os.path.join(output_dir, pop_filename), index=False)
            logger.info(f"✅ Population RINPERSOON-YEAR-BUCKET saved to '{pop_filename}'")



        # --------------- SAMPLES FOR KS/WASSERSTEIN -----------------
        pop_samples = expand_counts_to_samples(pop_counts)
        if not cfg.get(LISS_FILE, 0) is None:
            liss_samples = expand_counts_to_samples(liss_counts)

        # --------------- METRIC COMPUTATION -----------------
        if not cfg.get(LISS_FILE, 0) is None:
            pearson_corr = compute_pearson(pop_probs, liss_probs)
            ks_div = compute_ks_from_samples(pop_samples, liss_samples)
            js_div = compute_js(pop_probs, liss_probs)
            kl_div = compute_kl(pop_probs, liss_probs)
            chi_square_stat = compute_chi_square(pop_counts, liss_counts)
            wasserstein_dist = compute_wasserstein(pop_samples, liss_samples)
            hellinger_dist = compute_hellinger(pop_probs, liss_probs)
            cosine_sim = compute_cosine_similarity(pop_probs, liss_probs)

        pop_entropy = compute_entropy(pop_probs)
        pop_gini = compute_gini(pop_counts)
        if not cfg.get(LISS_FILE, 0) is None:
            liss_entropy = compute_entropy(liss_probs)
            liss_gini = compute_gini(liss_counts)

        # --------------- COVERAGE -----------------
        coverage_threshold = 1e-6
        buckets_covered_pop = compute_coverage(pop_probs, threshold=coverage_threshold)
        if not cfg.get(LISS_FILE, 0) is None:
            buckets_covered_liss = compute_coverage(liss_probs, threshold=coverage_threshold)

        max_pct_pop = max(pop_probs)
        if not cfg.get(LISS_FILE, 0) is None:
            max_pct_liss = max(liss_probs)

        # --------------- FRACTIONAL COVERAGE -----------------
        buckets_to_cover_90_pop = buckets_to_cover_fraction(pop_probs, 0.90)
        if not cfg.get(LISS_FILE, 0) is None:
            buckets_to_cover_90_liss = buckets_to_cover_fraction(liss_probs, 0.90)

        if cfg.get(LISS_FILE, 0) is None:
            return {
            'embedding_type': emb_type,
            'year': year,
            'num_buckets': num_buckets,
            'pop_entropy': pop_entropy,
            'pop_gini': pop_gini,
            'max_pct_pop': max_pct_pop,
            'buckets_covered_pop': buckets_covered_pop,
            'buckets_to_cover_90_pop': buckets_to_cover_90_pop,
        }

        # --------------- SAVE RESULTS -----------------
        return {
            'embedding_type': emb_type,
            'year': year,
            'num_buckets': num_buckets,
            'samples': sample,
            'pearson_corr': pearson_corr,
            'ks_div': ks_div,
            'js_div': js_div,
            'kl_div': kl_div,
            'chi_square': chi_square_stat,
            'wasserstein': wasserstein_dist,
            'hellinger': hellinger_dist,
            'cosine_sim': cosine_sim,
            'pop_entropy': pop_entropy,
            'liss_entropy': liss_entropy,
            'pop_gini': pop_gini,
            'liss_gini': liss_gini,
            'max_pct_pop': max_pct_pop,
            'max_pct_liss': max_pct_liss,
            'buckets_covered_pop': buckets_covered_pop,
            'buckets_covered_liss': buckets_covered_liss,
            'buckets_to_cover_90_pop': buckets_to_cover_90_pop,
            'buckets_to_cover_90_liss': buckets_to_cover_90_liss
        }
    except Exception as e:
        logger.error(f"❌ Failed for {embedding_row.embedding_name}, {embedding_row.year}, {num_buckets}: {e}", exc_info=True)
        return None

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfgfile", help="Path to your JSON config")
    parser.add_argument("--index", type=int, default=None, help="Array task index (Slurm) or None for serial")
    args = parser.parse_args()

    # Load config
    try:
        logger.info(f"✅ Loading config: {args.cfgfile}")
        cfg = read_json(args.cfgfile)
    except Exception as e:
        logger.error(f"❌ Failed to load config file: {e}")
        sys.exit(1)

    if cfg.get(MERGE_ALL_EMBEDDING_FILES, 0):
        pass
    else:  
    # Build param grid
        try:
            param_grid = build_param_grid(cfg)
            total_jobs = len(param_grid)
            logger.info(f"✅ Parameter grid has {total_jobs} jobs.")
        except Exception as e:
            logger.error(f"❌ Failed to build parameter grid: {e}")
            sys.exit(1)

    # Decide on array or serial mode
    if args.index is None:
        results = []
        logger.info("✅ No --index given: running ALL jobs in SERIAL mode with merged embedding files")
        if cfg.get(MERGE_ALL_EMBEDDING_FILES, 0):
            try:
                # Create embedding row
                embedding_row = SimpleNamespace()
                embedding_row.embedding_name = 'merged_embeddings'
                embedding_row.year = 'all'
                embedding_row.file_path = cfg[EMBEDDING_FILE]

                num_buckets = cfg[NUM_BUCKETS_LIST][0]  # Use first bucket size
                sample = 'liss-people'  # Default sample set

                results = process_embedding(embedding_row, num_buckets, sample, cfg)
            except Exception as e:
                logger.error(f"❌ Error processing merged embedding file: {e}")
                sys.exit(1)
        else:
            logger.info("✅ No --index given: running ALL jobs in SERIAL mode")
            for i, (embedding_row, num_buckets, sample) in enumerate(tqdm(param_grid, desc='Processing the embedding file',unit='file')):
                try:
                    logger.info(f"▶️ [Serial] Processing job {i+1}/{total_jobs}: {embedding_row.embedding_name}")
                    res = process_embedding(embedding_row, num_buckets, sample, cfg)
                    if res is not None:
                        results.append(res)
                except Exception as e:
                    logger.error(f"❌ Error processing job {i}: {e}")
                    logger.error(traceback.format_exc())
                    continue

        # Save all results
        if results:
            try:
                output_dir = cfg[OUTPUT_DIR]
                os.makedirs(output_dir, exist_ok=True)
                df = pd.DataFrame(results)
                df.to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)
                logger.info("✅ All metrics saved to 'metrics_summary.csv'")
            except Exception as e:
                logger.error(f"❌ Failed to save metrics_summary.csv: {e}")
    else:
        # Array mode: run only one index
        i = args.index
        if i < 0 or i >= len(param_grid):
            logger.error(f"❌ Invalid index {i}. Must be between 0 and {len(param_grid)-1}. Exiting.")
            sys.exit(1)

        embedding_row, num_buckets, sample = param_grid[i]
        logger.info(f"✅ Running ARRAY mode for index {i}: {embedding_row.embedding_name}")

        try:
            result = process_embedding(embedding_row, num_buckets, sample, cfg)
            if result is not None:
                output_dir = cfg[OUTPUT_DIR]
                os.makedirs(output_dir, exist_ok=True)
                single_df = pd.DataFrame([result])
                outfile = os.path.join(output_dir, f"metrics_summary_part_{i}.csv")
                single_df.to_csv(outfile, index=False)
                logger.info(f"✅ Metrics for index {i} saved to '{outfile}'")
        except Exception as e:
            logger.error(f"❌ Failed to process index {i}: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)

    logger.info("✅ Script complete")