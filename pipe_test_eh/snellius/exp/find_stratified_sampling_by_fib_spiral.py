import os
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

# ----------------------------
# 1. Data loading
# ----------------------------
def load_data(embedding_file, background_file):
    print("✅ Loading embedding data...")
    df_emb = pd.read_parquet(embedding_file)
    print(f" - Rows: {len(df_emb)}, Columns: {list(df_emb.columns)}")

    print("✅ Loading background data...")
    df_bg = pd.read_parquet(background_file)
    print(f" - Rows: {len(df_bg)}, Columns: {list(df_bg.columns)}")

    # Ensure matching ID column names
    df_bg = df_bg.rename(columns={"RINPERSOON": "rinpersoon_id"})

    print("✅ Joining on rinpersoon_id...")
    df_merged = df_emb.merge(df_bg, on="rinpersoon_id", how="inner")
    print(f" - Joined rows: {len(df_merged)}")
    return df_merged

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

# deprecated
def generate_uniform_sphere(samples, dim):
    """
    Default strategy: uniformly sample unit vectors on hypersphere.
    """
    vecs = np.random.randn(samples, dim)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs

# deprecated
def generate_sphere(samples, dim, strategy="uniform", **kwargs):
    """
    Generalized cone center generator.
    
    Args:
        samples (int): Number of cone centers to generate.
        dim (int): Embedding dimension.
        strategy (str or callable): How to place centers.
            - "uniform": random directions on sphere.
            - callable: user-defined function.
        **kwargs: Extra arguments for custom strategy.
    
    Returns:
        ndarray of shape (samples, dim): Cone centers.
    """
    if strategy == "uniform":
        return generate_uniform_sphere(samples, dim)
    
    elif callable(strategy):
        return strategy(samples, dim, **kwargs)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
# deprecated
def density_weighted_strategy(samples, dim, data):
    """
    Example placeholder: cluster data, return cluster centers.
    """
    kmeans = KMeans(n_clusters=samples)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return centers

# ----------------------------
# 3. Cone assignment
# ----------------------------
def mydist(x, y, **kwargs):
    f = kwargs["f"]
    return np.sum(np.abs(x - y) ** f) ** (1. / f)


def nearest_nei_dist_mean(X, k, distance='cosine', f=None):
    if distance == 'cosine':
        A = kneighbors_graph(X=X, n_neighbors=k, metric='cosine', include_self=False)
        aver_dist_list = []
        for i in range(len(X)):
            x = np.asarray(X[i])
            aver_dist = []
            for j in A[i].indices:
                if i != j:
                    y = np.asarray(X[j])
                    dist = 1. - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
                    aver_dist.append(dist)
            aver_dist_list.append(np.mean(aver_dist))
        return aver_dist_list
    elif distance == 'l2_distance':
        A = kneighbors_graph(X=X, n_neighbors=k, metric='minkowski', p=2, include_self=False)
        aver_dist_list = []
        for i in range(len(X)):
            x = np.asarray(X[i])
            aver_dist = []
            for j in A[i].indices:
                if i != j:
                    aver_dist.append(np.linalg.norm(x - np.asarray(X[j])))
            aver_dist_list.append(np.mean(aver_dist))
        return aver_dist_list
    elif distance == 'l1_distance':
        A = kneighbors_graph(X=X, n_neighbors=k, metric='minkowski', p=1, include_self=False)
        aver_dist_list = []
        for i in range(len(X)):
            x = np.asarray(X[i])
            aver_dist = []
            for j in A[i].indices:
                if i != j:
                    aver_dist.append(np.sum(np.abs(x - np.asarray(X[j]))))
            aver_dist_list.append(np.mean(aver_dist))
        return aver_dist_list
    elif distance == 'frac_distance':
        p = len(X[0])
        if f is None:
            f = 1. / p
        A = kneighbors_graph(X=X, n_neighbors=k, metric=mydist, metric_params={"f": f}, include_self=False)
        aver_dist_list = []
        for i in range(len(X)):
            x = np.asarray(X[i])
            aver_dist = []
            for j in A[i].indices:
                if i != j:
                    aver_dist.append(np.sum(np.abs(x - np.asarray(X[j])) ** f) ** (1. / f))
            aver_dist_list.append(np.mean(aver_dist))
        return aver_dist_list


def nearest_nei_dist_std(X, k, distance='cosine'):
    if distance == 'cosine':
        A = kneighbors_graph(X=X, n_neighbors=k, metric='cosine', include_self=False)
        aver_dist_list = []
        for i in range(len(X)):
            x = np.asarray(X[i])
            aver_dist = []
            for j in A[i].indices:
                if i != j:
                    y = np.asarray(X[j])
                    dist = 1. - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
                    aver_dist.append(dist)
            aver_dist_list.append(np.std(aver_dist))
        return aver_dist_list
    elif distance == 'l2_distance':
        A = kneighbors_graph(X=X, n_neighbors=k, metric='minkowski', p=2, include_self=False)
        aver_dist_list = []
        for i in range(len(X)):
            x = np.asarray(X[i])
            aver_dist = []
            for j in A[i].indices:
                if i != j:
                    y = np.asarray(X[j])
                    dist = np.linalg.norm(x - y)
                    aver_dist.append(dist)
            aver_dist_list.append(np.std(aver_dist))
        return aver_dist_list
    elif distance == 'l1_distance':
        A = kneighbors_graph(X=X, n_neighbors=k, metric='minkowski', p=1, include_self=False)
        aver_dist_list = []
        for i in range(len(X)):
            x = np.asarray(X[i])
            aver_dist = []
            for j in A[i].indices:
                if i != j:
                    aver_dist.append(np.sum(np.abs(x - np.asarray(X[j]))))
            aver_dist_list.append(np.std(aver_dist))
        return aver_dist_list
    
# deprecated
def assign_points_to_cones(embeddings, cone_centers):
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cone_centers = cone_centers / np.linalg.norm(cone_centers, axis=1, keepdims=True)
    similarities = embeddings @ cone_centers.T
    return np.argmax(similarities, axis=1)

# ----------------------------
# 4. Sampling evenly from cones
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

def bucket_sampling(words: List[str],
                    pos_tags: List[str],
                    bucket_ids: List[int],
                    k: int = 100) -> Tuple[List[str], List[str]]:
    """Sample k words proportionally from each bucket, handling rounding.
    
    Args:
      words: Original words in order.
      pos_tags: POS tags aligned with words.
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
    bucketed_pos = {}
    for i, bid in enumerate(bucket_ids):
        if bid not in bucketed_words:
            bucketed_words[bid] = []
            bucketed_pos[bid] = []
        bucketed_words[bid].append(words[i])
        bucketed_pos[bid].append(pos_tags[i])

    sampled_words = []
    sampled_pos = []
    for bid in bucketed_words:
        cnt = samples_per_bucket[bid]
        if cnt > 0:
            indices = list(range(len(bucketed_words[bid])))
            random.shuffle(indices)
            chosen = indices[:cnt]
            for idx in chosen:
                sampled_words.append(bucketed_words[bid][idx])
                sampled_pos.append(bucketed_pos[bid][idx])

    return sampled_words, sampled_pos

def sample_proportionally_from_buckets(df, bucket_col, sample_size):
    """
    Sample proportionally from each bucket to match the overall density distribution.
    """
    total = len(df)
    bucket_counts = df[bucket_col].value_counts().sort_index()
    buckets = bucket_counts.index

    # Determine how many samples per bucket (proportional to size)
    samples_per_bucket = {}
    for b in buckets:
        fraction = bucket_counts[b] / total
        samples_per_bucket[b] = max(1, round(sample_size * fraction))

    # Adjust rounding error
    diff = sample_size - sum(samples_per_bucket.values())
    if diff != 0:
        sorted_buckets = sorted(buckets, key=lambda x: bucket_counts[x], reverse=True)
        idx = 0
        while diff != 0 and idx < len(sorted_buckets):
            b = sorted_buckets[idx]
            if diff > 0:
                samples_per_bucket[b] += 1
                diff -= 1
            else:
                if samples_per_bucket[b] > 1:
                    samples_per_bucket[b] -= 1
                    diff += 1
            idx = (idx + 1) % len(sorted_buckets)

    # Actually sample
    sampled_rows = []
    for b in buckets:
        subset = df[df[bucket_col] == b]
        n_to_sample = samples_per_bucket[b]
        if len(subset) <= n_to_sample:
            sampled = subset
        else:
            sampled = subset.sample(n=n_to_sample, random_state=42)
        sampled_rows.append(sampled)

    final_sample = pd.concat(sampled_rows)
    print(f"✅ Sampled {len(final_sample)} rows total")
    return final_sample

# deprecated
def sample_evenly_from_cones(df, assignments, num_samples):
    df = df.copy()
    df["cone"] = assignments
    unique_cones = df["cone"].unique()
    num_cones = len(unique_cones)
    print(f"✅ Total cones: {num_cones}")
    print(df["cone"].value_counts())

    samples_per_cone = max(1, num_samples // num_cones)
    sampled_rows = []

    for cone in unique_cones:
        subset = df[df["cone"] == cone]
        if len(subset) <= samples_per_cone:
            sampled = subset
        else:
            sampled = subset.sample(samples_per_cone, random_state=42)
        sampled_rows.append(sampled)

    final_sample = pd.concat(sampled_rows)
    print(f"✅ Sampled {len(final_sample)} rows total")
    return final_sample

# ----------------------------
# 5. Compare metadata distribution
# ----------------------------
def compare_metadata_distribution(full_df, sample_df, variable):
    print(f"\n📊 Comparing distribution for: {variable}")

    full_counts = full_df[variable].value_counts(normalize=True).sort_index()
    sample_counts = sample_df[variable].value_counts(normalize=True).sort_index()

    comparison = pd.DataFrame({
        "Full": full_counts,
        "Sample": sample_counts
    }).fillna(0)
    comparison["AbsDiff"] = (comparison["Full"] - comparison["Sample"]).abs()

    print(comparison)
    return comparison

def save_comparison_to_csv(comparison_df, variable, output_file):
    comparison_df.index.name = variable
    comparison_df.reset_index().to_csv(output_file, index=False)
    print(f"✅ Comparison table saved to {output_file}")

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
    
    print(f"✅ Comparison plot saved to {output_file}")


# ----------------------------
# 6. MAIN
# ----------------------------
def main():
    import os

    # -------- File paths --------
    embedding_file = "/projects/0/prjs1019/data/fake_embs/feb20_test/feb20/mean.parquet"
    background_file = "/projects/0/prjs1019/data/fake_data_v0/step2/background.parquet"

    output_dir = "~/pipe_test_eh/snellius/exp/find_stratified_sampling_by_fib_spiral_output"
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # -------- Config --------
    dim = 160
    b = 100   # number of sphere directions / buckets
    k = 1000  # sample size
    variables_to_compare = ["gender", "year", "month", "municipality"]
    random.seed(42)
    np.random.seed(42)

    # -------- Load & join --------
    df = load_data(embedding_file, background_file)
    print(df.head())

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



# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    main()








