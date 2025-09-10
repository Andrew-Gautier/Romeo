#!/bin/bash

# Directory containing input files
INPUT_DIR="/path/to/input_files"
# Directory to deposit processed files
OUTPUT_DIR="/path/to/output_folder"
# Preprocessing script path
PREPROCESS_SCRIPT="/path/to/preprocess_script.sh"
# Number of files to process at a time
BATCH_SIZE=5

mkdir -p "$OUTPUT_DIR"

files=("$INPUT_DIR"/*)
total_files=${#files[@]}

for ((i=0; i<total_files; i+=BATCH_SIZE)); do
    batch=("${files[@]:i:BATCH_SIZE}")
    for file in "${batch[@]}"; do
        filename=$(basename "$file")
        "$PREPROCESS_SCRIPT" "$file" > "$OUTPUT_DIR/$filename.processed" &
    done
    wait
done