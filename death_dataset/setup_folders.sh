#!/bin/bash
# ============================================================================
# SETUP FOLDER STRUCTURE FOR MORTALITY LABELS
# ============================================================================
#
# This script creates the required folder structure for the mortality label
# generation. Run this BEFORE running the main label generation script.
#
# Usage:
#   ./setup_folders.sh /path/to/output/directory
#
# Example:
#   ./setup_folders.sh /projects/0/prjs1589/stonybrook/evaluation/labels/mortality
#
# ============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <output_directory>"
    echo ""
    echo "Example:"
    echo "  $0 /projects/0/prjs1589/stonybrook/evaluation/labels/mortality"
    exit 1
fi

OUTPUT_DIR="$1"

echo "Creating folder structure at: ${OUTPUT_DIR}"
echo ""

# Create main folders
mkdir -p "${OUTPUT_DIR}/all"
mkdir -p "${OUTPUT_DIR}/subset"
mkdir -p "${OUTPUT_DIR}/all-splits/train"
mkdir -p "${OUTPUT_DIR}/all-splits/val"
mkdir -p "${OUTPUT_DIR}/all-splits/test"
mkdir -p "${OUTPUT_DIR}/subset-splits/train"
mkdir -p "${OUTPUT_DIR}/subset-splits/val"
mkdir -p "${OUTPUT_DIR}/subset-splits/test"
mkdir -p "${OUTPUT_DIR}/stats/plots"

echo "Created folder structure:"
echo ""
find "${OUTPUT_DIR}" -type d | sort | sed "s|${OUTPUT_DIR}|.|g"

echo ""
echo "Done! You can now run the label generation script."
echo ""
echo "Example:"
echo "  sbatch run_mortality_labels.sh"
