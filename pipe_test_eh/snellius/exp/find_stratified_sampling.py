import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import random

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
def generate_uniform_sphere(samples, dim):
    """
    Default strategy: uniformly sample unit vectors on hypersphere.
    """
    vecs = np.random.randn(samples, dim)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs

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
def assign_points_to_cones(embeddings, cone_centers):
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cone_centers = cone_centers / np.linalg.norm(cone_centers, axis=1, keepdims=True)
    similarities = embeddings @ cone_centers.T
    return np.argmax(similarities, axis=1)

# ----------------------------
# 4. Sampling evenly from cones
# ----------------------------
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

# ----------------------------
# 6. MAIN
# ----------------------------
def main():
    # -------- File paths --------
    embedding_file = "/projects/0/prjs1019/data/fake_embs/feb20_test/feb20/mean.parquet"
    background_file = "/projects/0/prjs1019/data/fake_data_v0/step2/background.parquet"

    # -------- Config --------
    dim = 160
    num_cones = 100
    sample_size = 1000
    strategy = "density_weighted_strategy" 
    variable_to_compare = "gender"  # Or "year", "month", "municipality"
    random.seed(42)
    np.random.seed(42)

    # -------- Load & join --------
    df = load_data(embedding_file, background_file)
    print(df.head())

    # -------- Extract embeddings --------
    emb_cols = [c for c in df.columns if c.startswith("emb")]
    embeddings = df[emb_cols].to_numpy()

    # -------- Generate cone centers --------
    cone_centers = generate_sphere(num_cones, dim, strategy)
    print(f"✅ Generated {num_cones} cone centers in {dim}D space")

    # -------- Assign cones --------
    assignments = assign_points_to_cones(embeddings, cone_centers)
    print(f"✅ Assigned all people to cones")

    # -------- Sample evenly --------
    sample_df = sample_evenly_from_cones(df, assignments, sample_size)

    # -------- Compare distribution --------
    compare_metadata_distribution(df, sample_df, variable_to_compare)

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    main()








