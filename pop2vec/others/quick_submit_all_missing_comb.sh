#!/bin/bash

# 1. Validate Argument 1: The CSV file
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_missing_combinations_registry.csv> [optional_script_folder]"
    exit 1
fi

CSV_FILE="$1"

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file '$CSV_FILE' not found."
    exit 1
fi

# Get the directory where the CSV file resides
CSV_PARENT_DIR=$(dirname "$CSV_FILE")

# 2. Handle Argument 2: The slurm_scripts folder path
if [ -z "$2" ]; then
    # Default: look for slurm_scripts in the same folder as the CSV
    SCRIPT_DIR="$CSV_PARENT_DIR/slurm_scripts"
else
    if [[ "$2" = /* ]]; then
        # It is an absolute path
        SCRIPT_DIR="$2"
    else
        # It is a relative path; append to CSV parent dir
        SCRIPT_DIR="$CSV_PARENT_DIR/$2"
    fi
fi

# Verify script directory exists
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "Error: Script directory '$SCRIPT_DIR' does not exist."
    exit 1
fi

echo "Reading configs from: $CSV_FILE"
echo "Searching scripts in: $SCRIPT_DIR"

# 3. Extract the 'config' column from the CSV to a temporary file
# We use awk to find the column index named "config" dynamically, 
# then print that column for all rows > 1 (skipping header).
PATTERN_FILE=$(mktemp)
awk -F, '
    NR==1 {
        for (i=1; i<=NF; i++) {
            if ($i == "config") {
                col_idx = i
            }
        }
    }
    NR>1 && col_idx {
        print $col_idx
    }
' "$CSV_FILE" > "$PATTERN_FILE"

# Check if we actually found configs
if [ ! -s "$PATTERN_FILE" ]; then
    echo "Error: Could not find 'config' column or data in CSV."
    rm "$PATTERN_FILE"
    exit 1
fi

# 4. Iterate through files and submit if match found
cnt=0
# Iterate over .sh files in the determined directory
for file in "$SCRIPT_DIR"/*.sh; do
    # Check if file actually exists (handles empty directory case)
    [ -e "$file" ] || continue
    
    # grep flags:
    # -F: Fixed string (not regex)
    # -q: Quiet (don't output the match line, just return exit code)
    # -f: Take patterns from file (our list of configs)
    if grep -Fqf "$PATTERN_FILE" "$file"; then
        
        echo "Found matching config in: $file"
        echo "Submitting job..."
        
        # Execute the sbatch command
        sbatch "$file"
        
        echo "Done ----------------------------------------"
        cnt=$((cnt+1))
        echo "Processed $file, count= $cnt"
    fi
done

# Clean up temporary pattern file
rm "$PATTERN_FILE"