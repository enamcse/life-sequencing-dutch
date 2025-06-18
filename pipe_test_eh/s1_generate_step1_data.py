import pandas as pd
import random
import string
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from tqdm import tqdm

# Constants
MIN_ROWS = 100000#1_000_000
MAX_ROWS = 500000#10_000_000
GENESIS_DATE = pd.Timestamp('1971-12-30')

def generate_random_string(length=5):
    """Generate a random string of fixed length."""
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def generate_random_description(val):
    """Generate a random natural language description for a value."""
    descriptions = [
        f"Description for value {val}",
        f"This value represents {val}",
        f"Value {val} indicates a special case",
        f"An example of {val}",
        f"{val} is significant because...",
        f"The value {val} corresponds to an important category",
        f"Explanation for {val}",
        f"The code {val} stands for...",
        f"{val} is associated with certain characteristics",
        f"Note about value {val}"
    ]
    return random.choice(descriptions)

def generate_random_data(columns, num_rows, name):
    """Generate random data for given columns and number of rows."""
    global RANDOM_ID_POOL
    if 'background' in name: 
      num_rows = 10000000
    print(f'generating {num_rows} rows with {len(columns)} columns')
    data = {}
    current_ids = np.random.choice(RANDOM_ID_POOL, size=num_rows//8)
    for col in columns:
        if col == 'RINPERSOON':
            data[col] = np.random.choice(current_ids, size=num_rows)
        elif col == 'daysSinceFirstEvent':
            data[col] = np.random.randint(0, 18_000, size=num_rows)  # Random days since genesis (roughly 50 years)
        elif col == 'age':
            data[col] = np.random.randint(16, 100, size=num_rows)  # Age between 16 and 100
        elif col == 'month':
            data[col] = np.random.randint(1, 13, size=num_rows)
        elif col == 'year':
            data[col] = np.random.randint(1940, 2020, size=num_rows)
        elif col == 'gender':
            data[col] = np.random.randint(1, 3, size=num_rows)
        elif col == 'municipality':
            data[col] = np.random.randint(1, 500, size=num_rows)
        else:
            # Randomly decide if the column is numeric or categorical
            if random.choice([True, False]):  # 50% probability for each
                data[col] = np.random.randint(0, 1000, size=num_rows)  # Example numeric range
            else:
                data[col] = [generate_random_string(2) for _ in range(num_rows)]
    return pd.DataFrame(data)

def create_parquet_and_metadata(df, file_name):
    """Save DataFrame as Parquet and create corresponding metadata Parquet."""
    # Save data to Parquet
    parquet_path = f'step1/{file_name}.parquet'
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)
    
    # Generate metadata
    metadata = {
        'Name': [],
        'Type': [],
        'ValueLabels': []
    }
    for col in df.columns:
        if col in ['RINPERSOON', 'daysSinceFirstEvent', 'age']:
          continue
        metadata['Name'].append(col)
        col_type = 'Numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'String'
        metadata['Type'].append(col_type)
        x = random.randint(1, 10)
        unique_values = df[col].dropna().unique()
        if len(unique_values) > x:
            sampled_values = np.random.choice(unique_values, x, replace=False)
        else:
            sampled_values = unique_values
        value_labels = {}
        if 'background' not in file_name:
          for val in sampled_values:
              # Generate a random natural language description
              description = generate_random_description(val)
              value_labels[val] = description
        metadata['ValueLabels'].append(str(value_labels))
    
    metadata_df = pd.DataFrame(metadata)
    meta_parquet_path = f'step1/{file_name}_meta.parquet'
    meta_table = pa.Table.from_pandas(metadata_df)
    pq.write_table(meta_table, meta_parquet_path)
    
    return parquet_path, meta_parquet_path

def process_files(columns_dict):
    for name, columns in tqdm(columns_dict.items()):
        num_rows = random.randint(MIN_ROWS, MAX_ROWS)
        df = generate_random_data(columns, num_rows, name)
        create_parquet_and_metadata(df, name)

if __name__ == '__main__':
    dir_path = 'empty_files/'  # Replace with your actual directory path
    columns_dict = {}
    RINPERSOONS_PATH = 'fake_RINPERSOONS.csv'

    for file_name in os.listdir(dir_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(dir_path, file_name)
            df = pd.read_csv(file_path, nrows=0)  # Read only the header
            columns = df.columns.tolist()
            columns_dict[os.path.splitext(file_name)[0]] = columns
            

    RANDOM_ID_POOL = pd.read_csv(RINPERSOONS_PATH)['RINPERSOON'].unique()
    
    process_files(columns_dict)  # Execute the data generation and saving process

