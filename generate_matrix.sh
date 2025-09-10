#!/bin/bash
# Script to generate tensor sets for language testing matrix

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Set up the environment if it's not already activated
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check for arguments
SEED=${1:-42}
SAMPLES=${2:-500}

echo "Generating language testing matrix with seed $SEED and $SAMPLES samples per class"

# Run the Python script
python generate_language_matrix.py $SEED $SAMPLES

echo "Matrix generation completed!"
