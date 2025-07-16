import argparse
import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()]
    )

def main():
    setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_file", required=True, help="Path to the parquet file")
    parser.add_argument("--year1_list", type=int, nargs="+", required=True, help="List of starting years (e.g. 2013 2014)")
    parser.add_argument("--end_year", type=int, required=True, help="End year (inclusive)")
    parser.add_argument("--sample_size", type=int, default=1000000, help="Random samples to draw once and reuse")
    parser.add_argument("--output_file", required=True, help="Output CSV file with all results")
    args = parser.parse_args()

    logging.info(f"✅ Loading parquet file: {args.parquet_file}")
    df = pd.read_parquet(args.parquet_file, columns=['RINPERSOON', 'YEAR', 'BUCKET_ID'])
    logging.info(f"✅ Loaded dataframe with {len(df):,} rows")

    # Collect union of all years needed
    all_eval_years = set()
    for y1 in args.year1_list:
        all_eval_years.update(range(y1, args.end_year + 1))
    all_eval_years = sorted(all_eval_years)
    logging.info(f"✅ Union of evaluation years required: {all_eval_years}")

    # Filter to people with entries in ALL of these years
    logging.info("✅ Identifying people with complete data for ALL required years...")
    year_counts = df[df['YEAR'].isin(all_eval_years)].groupby('RINPERSOON')['YEAR'].nunique()
    full_people = year_counts[year_counts == len(all_eval_years)].index
    logging.info(f"✅ Found {len(full_people):,} people with complete data across all years.")

    if len(full_people) == 0:
        logging.error("❌ No people with entries in all required years! Exiting.")
        return

    # Filter dataset to only those people
    df_full = df[df['RINPERSOON'].isin(full_people)]
    logging.info(f"✅ Filtered dataframe size: {len(df_full):,} rows")

    # Pivot once
    logging.info("✅ Creating pivot table for fast bucket lookup...")
    pivot = df_full.pivot(index='RINPERSOON', columns='YEAR', values='BUCKET_ID')
    logging.info(f"✅ Initial pivot shape (before dropping NAs): {pivot.shape}")

    # Drop rows with any missing years
    pivot = pivot.dropna()
    logging.info(f"✅ Pivot shape after dropping incomplete rows: {pivot.shape}")

    # Choose random sample of pairs ONCE
    logging.info(f"✅ Sampling {args.sample_size:,} random person pairs ONCE...")
    full_people_list = pivot.index.tolist()
    rng = np.random.default_rng(seed=42)
    random_pairs = []

    # Fast vectorized sampling of distinct pairs
    logging.info(f"✅ Sampling {args.sample_size:,} random person pairs ONCE (vectorized)...")

    N = len(full_people_list)
    if N < 2:
        logging.error("❌ Not enough people to sample distinct pairs!")
        return

    # Oversample a bit to account for filtering same-index
    oversample_factor = 1.2
    num_attempts = int(args.sample_size * oversample_factor)

    while True:
        a_indices = rng.integers(0, N, size=num_attempts)
        b_indices = rng.integers(0, N, size=num_attempts)
        mask = a_indices != b_indices
        valid_a = a_indices[mask]
        valid_b = b_indices[mask]
        if len(valid_a) >= args.sample_size:
            break
        oversample_factor *= 1.5
        num_attempts = int(args.sample_size * oversample_factor)
        logging.info(f"✅ Resampling with increased size: {num_attempts}")

    # Take exactly sample_size
    random_pairs = list(zip(
        [full_people_list[i] for i in valid_a[:args.sample_size]],
        [full_people_list[i] for i in valid_b[:args.sample_size]]
    ))

    logging.info(f"✅ Finished sampling {len(random_pairs):,} random pairs.")
    logging.info("✅ Random pairs sampling complete. Will reuse this set for all year pairs.")

    # Determine all year-pairs to evaluate
    all_pairs = []
    for y1 in args.year1_list:
        for y2 in range(y1 + 1, args.end_year + 1):
            all_pairs.append((y1, y2))

    logging.info(f"✅ Total number of year pairs to evaluate: {len(all_pairs)}")

    # Evaluate
    results = []
    for year1, year2 in tqdm(all_pairs, desc="Evaluating year pairs"):
        logging.info(f"▶️ Evaluating pair: ({year1}, {year2})")

        # SAME-person analysis
        both_years = pivot[[year1, year2]]
        total_same_person = len(both_years)
        if total_same_person > 0:
            same_matches = (both_years[year1] == both_years[year2]).sum()
            x_prob = same_matches / total_same_person
            logging.info(f"✅ Same-person: {same_matches}/{total_same_person} matches, x={x_prob:.6f}")
        else:
            x_prob = np.nan
            logging.warning(f"⚠️ No overlapping people for same-person analysis for ({year1}, {year2})")

        # RANDOM-person analysis using *same* random_pairs
        random_matches = 0
        rand_pairs_sampled = 0
        for a, b in random_pairs:
            b1 = pivot.at[a, year1] if year1 in pivot.columns else np.nan
            b2 = pivot.at[b, year2] if year2 in pivot.columns else np.nan
            if pd.notnull(b1) and pd.notnull(b2):
                rand_pairs_sampled += 1
                if b1 == b2:
                    random_matches += 1

        if rand_pairs_sampled > 0:
            y_prob = random_matches / rand_pairs_sampled
            logging.info(f"✅ Random-person: {random_matches}/{rand_pairs_sampled} matches, y={y_prob:.6f}")
        else:
            y_prob = np.nan
            logging.warning(f"⚠️ No valid random pairs for random-person analysis for ({year1}, {year2})")

        ratio = x_prob / y_prob if y_prob else np.nan

        results.append({
            "Year1": year1,
            "Year2": year2,
            "Same-person probability (x)": x_prob,
            "Random-person probability (y)": y_prob,
            "x/y": ratio,
            "Same-person pairs": total_same_person,
            "Random-person pairs": rand_pairs_sampled,
            "Sample size": args.sample_size
        })

    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    results_df.to_csv(args.output_file, index=False)
    logging.info(f"🎯 All evaluations complete! Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
