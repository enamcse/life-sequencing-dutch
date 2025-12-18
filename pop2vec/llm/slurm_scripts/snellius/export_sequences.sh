#!/bin/bash
#
#SBATCH --job-name=export_sequences
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH -p rome
#SBATCH -e /projects/0/prjs1589/stonybrook/logs/%x-%j.err
#SBATCH -o /projects/0/prjs1589/stonybrook/logs/%x-%j.out

echo "Job started on $(date)"
date

cd ~/life-sequencing-dutch/
source requirements/load_venv.sh

# Configuration - modify these paths as needed
H5_PATH="/projects/0/prjs1589/stonybrook/fake_data_v0/step5/encoding=nomlm/encoded.h5"
VOCAB_PATH="/projects/0/prjs1589/stonybrook/fake_data_v0/step5/vocab_v0.csv"
OUTPUT_DIR="/projects/0/prjs1589/stonybrook/llm/gen_out/sequence_exports"
FILE_PREFIX="person_without_birthday_tokens"

# Export settings
SEQUENCE_IDS="0-5"  # Export first 5 sequences (change as needed)
SEPARATE_FILES="--separate-files"  # Comment out for single file
INCLUDE_PADDING=""  # Add "--include-padding" to include padding tokens
MLM_ENCODED=""  # Add "--mlm-encoded" if data is MLM encoded
EXPORT_VOCAB_STATS="--export-vocab-stats"  # Comment out to skip vocab stats

echo "=== Sequence Export Configuration ==="
echo "HDF5 Path:        $H5_PATH"
echo "Vocabulary Path:  $VOCAB_PATH"
echo "Sequence IDs:     $SEQUENCE_IDS"
echo "Output Dir:       $OUTPUT_DIR"
echo "File Prefix:      $FILE_PREFIX"
echo "====================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the export
echo "Starting sequence export..."
date

python -m pop2vec.llm.src.new_code.export_sequences_to_csv \
    --h5-path "$H5_PATH" \
    --vocab-path "$VOCAB_PATH" \
    --sequence-ids "$SEQUENCE_IDS" \
    --output-dir "$OUTPUT_DIR" \
    --file-prefix "$FILE_PREFIX" \
    $SEPARATE_FILES \
    $INCLUDE_PADDING \
    $MLM_ENCODED \
    $EXPORT_VOCAB_STATS

if [ $? -eq 0 ]; then
    echo "✅ Export completed successfully!"
    echo "Output files saved in: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"
else
    echo "❌ Export failed!"
    exit 1
fi

date
echo "Job ended successfully"
