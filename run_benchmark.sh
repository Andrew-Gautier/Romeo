#!/bin/bash -lT
#SBATCH -J gpu_benchmark
#SBATCH --output=logs/benchmark_%j.log
#SBATCH --error=logs/benchmark_%j.err
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH -p gpu_7day
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00

# ============================================================================
# GPU Benchmark Script
# ============================================================================
# This script benchmarks single-GPU vs multi-GPU (DataParallel) performance
# to help diagnose training speed issues.
# ============================================================================

set -e

# Configuration
BASE_DIR="/scratch/aeg00011"
EXPERIMENT_NAME="Experiment_1"
TENSOR_DIR="${BASE_DIR}/${EXPERIMENT_NAME}"
WEIGHTS_PATH="${BASE_DIR}/aix3-7b-base (1).pt"

# Benchmark parameters
BATCH_SIZE=8
NUM_BATCHES=100  # Number of batches to benchmark (more = more accurate, but slower)

# Create logs directory
mkdir -p logs

# Activate conda environment
echo "============================================================"
echo "Activating conda environment..."
echo "============================================================"
conda activate python3112

# GPU diagnostics
echo ""
echo "============================================================"
echo "GPU Diagnostic Information"
echo "============================================================"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo "============================================================"

# Find k=1 dataset
echo ""
echo "============================================================"
echo "Finding k=1 dataset..."
echo "============================================================"
DATA_DIR=$(find "${TENSOR_DIR}" -maxdepth 1 -type d -name "juliet_c_simhash_k=1_*_seed42" | head -n 1)

if [ -z "${DATA_DIR}" ]; then
    echo "ERROR: Could not find k=1 dataset in ${TENSOR_DIR}"
    exit 1
fi

echo "Using dataset: ${DATA_DIR}"

# Run benchmark
echo ""
echo "============================================================"
echo "Running GPU Benchmark"
echo "============================================================"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Number of batches: ${NUM_BATCHES}"
echo "  Effective batch size (8 GPUs): $((BATCH_SIZE * 8))"
echo ""

python "${BASE_DIR}/benchmark_gpu.py" \
    --data-dir "${DATA_DIR}" \
    --weights "${WEIGHTS_PATH}" \
    --batch-size ${BATCH_SIZE} \
    --num-batches ${NUM_BATCHES}

echo ""
echo "============================================================"
echo "Benchmark Complete"
echo "============================================================"
