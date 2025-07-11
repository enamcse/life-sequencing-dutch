# name: medical_sav_to_parquet_convert_ossc.py
import pandas as pd
import pyreadstat
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

def convert_sav_to_parquet(sav_file, start_column, end_column, output_dir='.'):
    # Step 1: Read the .sav file using pyreadstat
    df, meta = pyreadstat.read_sav(sav_file)
    
    # Step 2: Split the events into start and end, and create a new column for event dates
    start_df = df[['RINPERSOON'] + [start_column]].copy()
    end_df = df[['RINPERSOON'] + [end_column]].copy()
    
    # Create an event indicator and date column
    start_df['event_date'] = pd.to_datetime(start_df[start_column], errors='coerce')
    start_df['begOrEnd'] = 1  # 1 for start
    
    end_df['event_date'] = pd.to_datetime(end_df[end_column], errors='coerce')
    end_df['begOrEnd'] = 2  # 2 for end
    
    # Merge start and end dataframes
    events_df = pd.concat([start_df[['RINPERSOON', 'event_date', 'begOrEnd']],
                           end_df[['RINPERSOON', 'event_date', 'begOrEnd']]], ignore_index=True)

    # Add 'age' and 'daySinceFirstEvent' columns
    events_df['age'] = (events_df['event_date'].apply(lambda x: (x - datetime(1971, 12, 30)).days) / 365.25)
    events_df['daySinceFirstEvent'] = events_df['event_date'].apply(lambda x: (x - datetime(1971, 12, 30)).days)
    
    # Step 3: Write the events dataframe to a parquet file (x.parquet)
    events_table = pa.Table.from_pandas(events_df)
    pq.write_table(events_table, f'{output_dir}/{sav_file.split("/")[-1].split(".")[0]}.parquet')

    # Step 4: Create the metadata file (x_meta.parquet)
    meta_rows = []
    for column in meta.column_names:
        # Infer type from DataFrame
        dtype = df[column].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            col_type = 'Numeric'
        else:
            col_type = 'String'

        # Get value labels if any
        value_labels_dict = meta.column_value_labels.get(column, {})
        meta_rows.append({
            'Name': column,
            'Type': col_type,
            'ValueLabels': value_labels_dict
        })
    
    meta_df = pd.DataFrame(meta_rows)
    meta_table = pa.Table.from_pandas(meta_df)
    pq.write_table(meta_table, f'{output_dir}/{sav_file.split("/")[-1].split(".")[0]}_meta.parquet')

    print(f"Conversion completed. Files saved as: {sav_file.split('/')[-1].split('.')[0]}.parquet and {sav_file.split('/')[-1].split('.')[0]}_meta.parquet")

# Usage:
sav_file = '/gpfs/ostor/ossc9424/data/health/MSZPrestatiesVEKT2023TABV1.sav'
start_column = 'VEKTMSZBegindatumPrest'  # Replace with the actual start date column name
end_column = 'VEKTMSZEinddatumPrest'  # Replace with the actual end date column name
output_dir = '/gpfs/ostor/ossc9424/data/eh/health_data'  # Specify your output directory

convert_sav_to_parquet(sav_file, start_column, end_column, output_dir)
