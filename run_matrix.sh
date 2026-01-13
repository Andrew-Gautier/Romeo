#!/bin/bash -lT

#SBATCH -J Lang_results
#SBATCH --output=30kperclassexperiment.log
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -n 1
#SBATCH -p inter_a100
#SBATCH --gpus 8


conda activate python3112
echo "============================================================"
echo "Starting Multilingual Vulnerability Detection Experiment"
echo "============================================================"

# Create necessary directories if they don't exist
mkdir -p models results plots checkpoints

# Run the experiment
echo "Running the experiment..."
python run_lang_matrix.py 

echo "============================================================"
echo "Experiment completed!"
echo "============================================================"

# Find the latest results file
latest=$(ls -t results/performance_matrix_*.csv 2>/dev/null | head -n 1)

if [ -n "$latest" ]; then
  echo "Latest results:"
  cat "$latest"
else
  echo "No results file found."
fi

echo "============================================================"
echo "Check the 'results' directory for detailed results and"
echo "the 'plots' directory for performance visualizations."
echo "============================================================"

read -p "Press Enter to continue..."
