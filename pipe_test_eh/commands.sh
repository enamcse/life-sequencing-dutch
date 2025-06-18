# Generate Step 1 data: Run from pipe_test_eh directory
# SBU AI cluster gave error due to capitalization of the file name fake_rinpersoon.csv but my local system did not.
mkdir step1
python s1_generate_step1_data.py

# Generate Step 2 data
python s2_subset_data_by_columns.py step1 step2 s1_to_s2_column_list.txt

# Generate Step 3 data: Run from life-sequence-dutch directory
sbatch s3_preprocess.sh
