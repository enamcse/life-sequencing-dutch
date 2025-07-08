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
    pq.write_table(events_table, f'{output_dir}/{sav_file.split(".")[0]}.parquet')

    # Step 4: Create the metadata file (x_meta.parquet)
    meta_rows = []
    for column, column_type in zip(meta.column_names, meta.column_types):
        column_values = list(meta.column_value_labels.get(column, {}).values())
        meta_rows.append({
            'Name': column,
            'Type': 'Numeric' if column_type == 'numeric' else 'String',
            'ValueLabels': column_values
        })
    
    meta_df = pd.DataFrame(meta_rows)
    meta_table = pa.Table.from_pandas(meta_df)
    pq.write_table(meta_table, f'{output_dir}/{sav_file.split(".")[0]}_meta.parquet')

    print(f"Conversion completed. Files saved as: {sav_file.split('.')[0]}.parquet and {sav_file.split('.')[0]}_meta.parquet")

# Example Usage:
sav_file = 'x.sav'
start_column = 'start_date_column_name'  # Replace with the actual start date column name
end_column = 'end_date_column_name'  # Replace with the actual end date column name

convert_sav_to_parquet(sav_file, start_column, end_column)
