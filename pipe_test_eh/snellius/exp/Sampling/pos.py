# Copyright 2025, Your Name
# This code follows Google's Python Style Guide, with max 80-char line length.
from nltk.corpus import wordnet
from tqdm import tqdm
import random
print("nello")
import spacy
print("hello")
import numpy as np
import nltk
from nltk.corpus import words as nltk_corpus

from typing import List, Tuple, Dict
from collections import Counter
from math import sqrt
from itertools import count
from copy import deepcopy

# We assume the following functions (uniform_sphere, etc.) come from
# previously provided code. For clarity, they are included here directly.

from math import cos, gamma, pi, sin
from typing import Callable, Iterator
from sklearn.neighbors import kneighbors_graph

import csv

def save_pos_table_to_csv(pos_table: List[List[str]], file_name: str) -> None:
    """Save the POS comparison table to a CSV file.
    
    Args:
      pos_table: The table with POS comparisons (list of lists).
      file_name: Name of the file to save the table.
    """
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["POS", "Original %", "Sampled %"])
        # Write rows
        writer.writerows(pos_table)

###############################################################################
# Provided code snippet for fibonacci-like sphere generation (uniform_sphere),
# plus related helpers (int_sin_m, primes, inverse_increasing).
###############################################################################
def mydist(x, y, **kwargs):
    """L^f distance for kneighbors_graph with fractional metric."""
    f = kwargs["f"]
    return np.sum(np.abs(x - y) ** f) ** (1.0 / f)

def int_sin_m(x: float, m: int) -> float:
    """Integral of sin^m(t) dt from 0 to x, computed recursively."""
    if m == 0:
        return x
    elif m == 1:
        return 1 - cos(x)
    else:
        return ((m - 1) / m * int_sin_m(x, m - 2)
                - cos(x) * sin(x) ** (m - 1) / m)

def primes() -> Iterator[int]:
    """Generate prime numbers infinitely."""
    yield from (2, 3, 5, 7)
    composites = {}
    ps = primes()
    next(ps)  # Skip 2
    p = next(ps)  # Should be 3
    assert p == 3
    psq = p * p
    for i in count(9, 2):
        if i in composites:
            step = composites.pop(i)
        elif i < psq:
            yield i
            continue
        else:
            assert i == psq
            step = 2 * p
            p = next(ps)
            psq = p * p
        i += step
        while i in composites:
            i += step
        composites[i] = step

def inverse_increasing(func: Callable[[float], float],
                       target: float,
                       lower: float,
                       upper: float,
                       atol: float = 1e-10) -> float:
    """Binary search for inverse of func, monotonic in [lower, upper]."""
    mid = (lower + upper) / 2
    val = func(mid)
    while abs(val - target) > atol:
        if val > target:
            upper = mid
        else:
            lower = mid
        mid = (upper + lower) / 2
        val = func(mid)
    return mid

def uniform_sphere(d: int, n: int) -> List[List[float]]:
    """Generate n points on the d-dimensional hypersphere."""
    assert d > 1
    assert n > 0
    pts = [[1 for _ in range(d)] for _ in range(n)]
    for i in range(n):
        t = 2 * pi * i / n
        pts[i][0] *= sin(t)
        pts[i][1] *= cos(t)
    for dim, prime_ in zip(range(2, d), primes()):
        offset = sqrt(prime_)
        mult = (gamma(dim / 2 + 0.5) /
                gamma(dim / 2) /
                sqrt(pi))

        def dim_func(y):
            return mult * int_sin_m(y, dim - 1)

        for i in range(n):
            deg = inverse_increasing(dim_func, i * offset % 1, 0, pi)
            for j in range(dim):
                pts[i][j] *= sin(deg)
            pts[i][dim] *= cos(deg)
    return pts
###############################################################################

def get_dictionary_words(n: int = 5000) -> List[str]:
    """Fetch n words from nltk corpus, ensuring length constraints.
    
    Args:
      n: Desired number of words.
      min_len: Minimum length of word to include.
      max_len: Maximum length of word to include.
    Returns:
      A list of words sampled from the English corpus.
    """
    nltk.download('words', quiet=True)
    all_words = [w for w in nltk_corpus.words()]
    # If there aren't enough words, pick what we can:
    if len(all_words) < n:
        chosen = all_words
    else:
        chosen = random.sample(all_words, n)
    return chosen

def filter_single_pos(words: List[str],
                      nlp_model: spacy.language.Language
                     ) -> Tuple[List[str], List[str]]:
    """Return only words with a single POS tag, dropping multiples.
    
    Args:
      words: List of words to analyze.
      nlp_model: A loaded spacy Language model.
    Returns:
      A tuple (filtered_words, pos_tags).
    """
    

    filtered_words = []
    pos_tags = []
    for w in tqdm(words):
        synsets = wordnet.synsets(w)
        pos_nltk = set(synset.pos() for synset in synsets)
        if len(pos_nltk) > 1:
          print(w, len(pos_nltk))
          continue
        doc = nlp_model(w)
        # If exactly 1 token, keep it
        if len(doc) == 1:
            filtered_words.append(w)
            pos_tags.append(doc[0].pos_)
            if len(filtered_words)%100 == 0:
              print(w)
    return filtered_words, pos_tags

def get_embeddings(words: List[str],
                   nlp_model: spacy.language.Language) -> np.ndarray:
    """Get normalized word embeddings from a spacy model."""
    embs = []
    for w in words:
        doc = nlp_model(w)
        # Single-token doc => doc[0].vector
        vec = doc[0].vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            embs.append(vec / norm)
        else:
            embs.append(vec)  # zero vector fallback
    return np.array(embs)

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

def create_pos_table(original_pos: List[str],
                     sampled_pos: List[str]) -> List[List[str]]:
    """Compute a table comparing original vs. sampled POS distributions.
    
    Args:
      original_pos: POS tags for entire set of words.
      sampled_pos: POS tags for sampled words.
    Returns:
      Table of shape 8 x 3: [pos_tag, orig% string, sample% string]
    """
    pos_counter_orig = Counter(original_pos)
    pos_counter_samp = Counter(sampled_pos)
    print(pos_counter_orig)
    common_pos = pos_counter_orig.most_common(9)
    total_orig = sum(pos_counter_orig.values())
    total_samp = sum(pos_counter_samp.values())

    table = []
    for pos_tag, _ in common_pos:
        orig_perc = (pos_counter_orig[pos_tag] / total_orig * 100
                     if total_orig else 0)
        samp_perc = (pos_counter_samp[pos_tag] / total_samp * 100
                     if total_samp else 0)
        table.append([
            pos_tag,
            f"{orig_perc:.2f}%",
            f"{samp_perc:.2f}%"
        ])
    return table

def save_embeddings(embeddings: np.ndarray, file_path: str) -> None:
    """Save embeddings as a NumPy binary file.

    Args:
      embeddings: Array of embeddings.
      file_path: Output .npy file path.
    """
    np.save(file_path, embeddings)

def load_embeddings(file_path: str) -> np.ndarray:
    """Load embeddings from a NumPy binary file.

    Args:
      file_path: Path to the .npy file.
    Returns:
      The loaded embeddings as a NumPy array.
    """
    return np.load(file_path)

def save_list(data: List[str], file_path: str) -> None:
    """Save a list of strings to a text file (one per line).

    Args:
      data: The list of strings to be saved.
      file_path: Output text file path.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(f"{item}\n")

def load_list(file_path: str) -> List[str]:
    """Load a list of strings from a text file.

    Args:
      file_path: Path to the text file.
    Returns:
      A list of strings (one per line).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def load_saved_data(emb_path: str,
                    words_path: str,
                    pos_path: str
                   ) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load embeddings, words, and POS tags from disk.

    Args:
      emb_path: Path to the .npy embeddings file.
      words_path: Path to the words text file.
      pos_path: Path to the POS tags text file.
    Returns:
      A tuple of (embeddings, words_filtered, pos_filtered).
    """
    embeddings = load_embeddings(emb_path)
    words = load_list(words_path)
    pos = load_list(pos_path)
    return embeddings, words, pos

def main():
    """Main function to demonstrate the workflow."""
    # Ensure NLTK 'words' corpus is downloaded in get_dictionary_words().
    # Load spaCy model (requires: python -m spacy download en_core_web_md)
    LOAD = True
    n, b, k = 50000, 30, 1000

    if LOAD:
      embeddings, words_filtered, pos_filtered = load_saved_data(
      "embeddings.npy", "words_filtered.txt", "pos_filtered.txt"
    )
    else:
      nlp = spacy.load("en_core_web_md")
      print("nlp loaded")
      # Step 1: Fetch real English words from nltk corpus
      n = 50000
      words_raw = get_dictionary_words(int(n*1.1))
      print(f"{len(words_raw)} words got from dictionary")
      # Step 2: Filter words to retain those with single POS
      words_filtered, pos_filtered = filter_single_pos(words_raw, nlp)
      print(f"words filtered, have {len(words_filtered)} words now")
      if n < len(words_filtered):
        # Generate a list of indices
        indices = list(range(len(words_filtered)))

        # Randomly sample n indices
        sampled_indices = random.sample(indices, n)

        # Use the indices to pick items from both lists
        words_filtered = [words_filtered[i] for i in sampled_indices]
        pos_filtered = [pos_filtered[i] for i in sampled_indices]
      
      print(f"words filtered is fixed. have {len(words_filtered)} words now")
      # Step 3: Get embeddings (normalized)
      embeddings = get_embeddings(words_filtered, nlp)
      save_embeddings(embeddings, "embeddings.npy")
      save_list(words_filtered, "words_filtered.txt")
      save_list(pos_filtered, "pos_filtered.txt")
      print("Embeddings, words_filtered, and pos_filtered saved to disk.")

    print(f"embeddings done")
    if embeddings.size == 0:
        print("No valid embeddings found. Exiting.")
        return
    d = embeddings.shape[1]
        # --- NEW PART: Save embeddings, words_filtered, pos_filtered to disk ---
    
    # If you want to reload later, you can call:
    # embeddings, words_filtered, pos_filtered = load_saved_data(
    #     "embeddings.npy", "words_filtered.txt", "pos_filtered.txt")

    # Step 4: Create b=100 points on the d-dim sphere
    sphere_pts = uniform_sphere(d, b)
    sphere_pts = np.array(sphere_pts)
    print("sphere points created")
    # Step 5: Bucketize all embeddings into b buckets via cosine sim
    bucket_ids = assign_buckets(embeddings, sphere_pts)
    print("buckets assigned")
    # Step 6: Sample k=100 words proportionally from each bucket
    k = 1000
    sampled_words, sampled_pos = bucket_sampling(words_filtered,
                                                 pos_filtered,
                                                 bucket_ids, k)

    print("words,pos sampled")
    # Step 7: Create a table with the 8 most common POS
    pos_table = create_pos_table(pos_filtered, sampled_pos)

    print("POS Comparison Table (Top 8):")
    print("POS\tOriginal\tSampled")
    for row in pos_table:
        print(f"{row[0]}\t{row[1]}\t{row[2]}")

    output_file = f"pos_comparison_table_{n}_{b}_{k}.csv"
    save_pos_table_to_csv(pos_table, output_file)
if __name__ == "__main__":
    main()
