import pandas as pd
import os
import sys
from pathlib import Path
import json

def find_missing_combinations(registry_file_path: str|Path, result_file_path: str|Path):
    """
    Finds combinations present in the registry file but missing from the result file.
    Handles backward compatibility for registry files missing 'target_column'.

    Args:
        registry_file_path: Path to the CSV file containing all possible combinations.
        result_file_path: Path to the CSV file containing the generated combinations (results).
    """
    try:
        # Load the files
        registry_df = pd.read_csv(registry_file_path)
        result_df = pd.read_csv(result_file_path)
    except FileNotFoundError as e:
        print(f"Error: One of the files was not found: {e}")
        return
    except pd.errors.EmptyDataError as e:
        print(f"Error: One of the files is empty: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the files: {e}")
        return

    # --- Backward Compatibility: Expand/Fill registry if cols are missing ---
    # Check which columns are missing
    missing_target = 'target_column' not in registry_df.columns
    missing_lr = 'lr' not in registry_df.columns
    missing_batch = 'batch' not in registry_df.columns

    if missing_target or missing_lr or missing_batch:
        print("Info: Missing columns (target, lr, or batch) in registry. Fetching from config files...")
        
        expanded_rows = []
        # Ensure we have a directory path (handle str or Path input)
        registry_path_obj = Path(registry_file_path)
        registry_dir = registry_path_obj.parent

        for _, row in registry_df.iterrows():
            config_rel_path = row.get('config')
            if pd.isna(config_rel_path):
                continue
            
            # Construct the full path to the config json
            config_path = registry_dir / str(config_rel_path)

            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Determine LR and Batch values
                # If column was missing, get from config. Else, use existing row value.
                row_lr = config_data.get('LR') if missing_lr else row.get('lr')
                row_batch = config_data.get('BATCH_SIZE') if missing_batch else row.get('batch')

                # If target_column is missing, we expand the row for each target in config
                if missing_target:
                    targets_dict = config_data.get('target_column', {})
                    if isinstance(targets_dict, dict):
                        for target in targets_dict.keys():
                            new_row = row.to_dict()
                            new_row['target_column'] = target
                            new_row['lr'] = row_lr
                            new_row['batch'] = row_batch
                            expanded_rows.append(new_row)
                else:
                    # Target exists, just update the row with fetched LR/Batch
                    new_row = row.to_dict()
                    new_row['lr'] = row_lr
                    new_row['batch'] = row_batch
                    expanded_rows.append(new_row)

            except Exception as e:
                print(f"Warning: Could not read/parse config file '{config_path}': {e}")
                continue

        if not expanded_rows:
            print("Error: Failed to expand/fill rows. Check config file paths and format.")
            return

        # Recreate registry_df with the expanded/filled data
        registry_df = pd.DataFrame(expanded_rows)
        print(f"Info: Registry processed. New shape: {registry_df.shape}")

    # --- Standard Logic ---


    # Define the columns for the unique combination in each file
    # Registry file columns (provided by user): task, model, target_column, lr, batch
    registry_cols = ['task', 'model', 'target_column', 'lr', 'batch']
    # Result file columns (provided by user): task_file, model_name, target, LR, BATCH-SIZE
    result_cols = ['task_file', 'model_name', 'target', 'LR', 'BATCH-SIZE']

    # --- Data Cleaning and Standardization ---
    try:
        # Convert numerical columns to a consistent type for reliable comparison
        registry_df['lr'] = pd.to_numeric(registry_df['lr'], errors='coerce').fillna(-1)
        registry_df['batch'] = pd.to_numeric(registry_df['batch'], errors='coerce').fillna(-1)
        result_df['LR'] = pd.to_numeric(result_df['LR'], errors='coerce').fillna(-1)
        result_df['BATCH-SIZE'] = pd.to_numeric(result_df['BATCH-SIZE'], errors='coerce').fillna(-1)
    except KeyError as e:
        print(f"Error: A required column for conversion is missing. Check column names: {e}")
        return

    # 1. Rename result columns to match registry columns
    rename_mapping = dict(zip(result_cols, registry_cols))
    result_renamed_df = result_df[result_cols].rename(columns=rename_mapping)

    # 2. Create a unique, hashable key for each combination (CRITICAL for matching floats like LR)
    def create_key(row):
        # Format 'lr' (float) to a specific decimal precision (e.g., 4) to ensure stable comparison
        # Convert 'batch' (integer) to string
        return (
            str(row['task']),
            str(row['model']),
            str(row['target_column']),
            f"{row['lr']:.4f}",
            str(int(row['batch']))
        )

    # Apply the key creation function to both dataframes
    registry_df['key'] = registry_df[registry_cols].apply(create_key, axis=1)
    result_renamed_df['key'] = result_renamed_df[registry_cols].apply(create_key, axis=1)

    # 3. Find missing keys using set difference
    registry_keys = set(registry_df['key'])
    result_keys = set(result_renamed_df['key'])
    missing_keys = registry_keys - result_keys

    # 4. Filter the original registry dataframe
    missing_combinations_df = registry_df[registry_df['key'].isin(missing_keys)].copy()

    # Drop the temporary 'key' column before saving
    missing_combinations_df = missing_combinations_df.drop(columns=['key'])

    print(f"Found {len(missing_combinations_df)} missing combinations.")

    if not missing_combinations_df.empty:
        # 5. Determine the output file path
        registry_dir = registry_file_path.parent
        output_file_name = f"missing_combinations_{registry_file_path.name}"
        output_file_path = registry_dir / output_file_name

        # 6. Save the missing combinations
        missing_combinations_df.to_csv(output_file_path, index=False)

        print(f"The missing combinations have been saved to: {output_file_path}")

    # 7. Find EXTRA rows (Foreign combinations OR Duplicates)
    # Condition: Key is NOT in registry OR Key is a Duplicate within the results
    extra_mask = ~result_renamed_df['key'].isin(registry_keys) | result_renamed_df.duplicated(subset='key')
    
    extra_rows_df = result_df[extra_mask]

    print(f"Found {len(extra_rows_df)} extra rows (foreign or duplicate) in result file.")

    if not extra_rows_df.empty:
        registry_dir = registry_file_path.parent
        extra_file_name = f"extra_combinations_{result_file_path.name}"
        extra_file_path = registry_dir / extra_file_name

        extra_rows_df.to_csv(extra_file_path, index=False)
        
        print(f"The extra rows have been saved to: {extra_file_path}")

# --- Example Usage (Replace 'registry.csv' and 'combined-static-evals.csv' with your actual file names) ---
# find_missing_combinations('registry.csv', 'combined-static-evals.csv')

def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python report_remaining_eval_results.py <registry_file_path> <result_file_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    registry_file = Path(sys.argv[1])
    result_file = Path(sys.argv[2])
    find_missing_combinations(registry_file, result_file)

if __name__ == "__main__":
    main()
