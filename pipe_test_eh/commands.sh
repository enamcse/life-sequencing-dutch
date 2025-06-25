# Generate Step 1 data: Run from pipe_test_eh directory
# SBU AI cluster gave error due to capitalization of the file name fake_rinpersoon.csv but my local system did not.
# mkdir step1
python pipe_test_eh/s1_generate_step1_data.py

# Generate Step 2 data
python pipe_test_eh/s2_subset_data_by_columns.py step1 step2 s1_to_s2_column_list.txt

# Generate Step 3 data
sbatch pipe_test_eh/s3_preprocess.sh

# Generate Step 4 data
sbatch pipe_test_eh/s4_create_parquet_seq.sh

# Generate Step 5 data
sbatch pipe_test_eh/s5_pipeline.sh

# Generate Step 6 data
sbatch pipe_test_eh/s6_pretrain_small.sh