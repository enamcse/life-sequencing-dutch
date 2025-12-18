#!/bin/bash
#
# Master script to repair, analyze, and evaluate generative model outputs
#
# Usage:
#   ./run_full_analysis.sh <input_tokens_file> <output_base_dir>
#
# Example:
#   ./run_full_analysis.sh \
#       /projects/0/prjs1589/stonybrook/llm/gen_out/pretty_tokens_20251212.txt \
#       /projects/0/prjs1589/stonybrook/llm/gen_out/full_analysis_20251212

set -e  # Exit on error

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_tokens_file> <output_base_dir>"
    echo ""
    echo "Example:"
    echo "  $0 /projects/0/prjs1589/stonybrook/llm/gen_out/pretty_tokens_20251212.txt \\"
    echo "     /projects/0/prjs1589/stonybrook/llm/gen_out/full_analysis_20251212"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_BASE="$2"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "GENERATIVE MODEL ANALYSIS PIPELINE"
echo "=============================================="
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_BASE"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Step 1: Repair the file
echo "STEP 1: Repairing output file..."
echo "----------------------------------------------"
REPAIRED_FILE="$OUTPUT_BASE/repaired_tokens.txt"
python "$SCRIPT_DIR/repair_generative_output.py" \
    "$INPUT_FILE" \
    "$REPAIRED_FILE"
echo ""

# Step 2: Analyze the repaired file
echo "STEP 2: Analyzing sequences..."
echo "----------------------------------------------"
ANALYSIS_DIR="$OUTPUT_BASE/analysis"
python "$SCRIPT_DIR/analyze_generative_output.py" \
    "$REPAIRED_FILE" \
    --output_dir "$ANALYSIS_DIR"
echo ""

# Step 3: Display summary
echo "=============================================="
echo "ANALYSIS COMPLETE"
echo "=============================================="
echo ""
echo "Output files:"
echo "  - Repaired tokens: $REPAIRED_FILE"
echo "  - Analysis results: $ANALYSIS_DIR/"
echo "    * summary.json"
echo "    * per_sequence_stats.csv"
echo "    * generated_category_dist.png"
echo "    * match_rates.png"
echo "    * diversity_metrics.png"
echo ""

# Display quick summary if jq is available
if command -v jq &> /dev/null; then
    echo "Quick Summary:"
    echo "----------------------------------------------"
    cat "$ANALYSIS_DIR/summary.json" | jq '{
        num_sequences,
        mean_token_match_rate,
        mean_category_match_rate,
        diversity_stats
    }'
else
    echo "Tip: Install 'jq' to see a quick summary here"
fi

echo ""
echo "Done!"
