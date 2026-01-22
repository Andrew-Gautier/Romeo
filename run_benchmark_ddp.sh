#!/bin/bash -lT
#SBATCH -J ddp_benchmark
#SBATCH --output=logs/benchmark_ddp_%j.log
#SBATCH --error=logs/benchmark_ddp_%j.err
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -n 1
#SBATCH -p gpu_7day
#SBATCH --gres=gpu:8
#SBATCH --time=00:30:00

# ============================================================================
# DDP Benchmark Script
# ============================================================================

set -e

BASE_DIR="/scratch/aeg00011"
EXPERIMENT_NAME="Experiment_1"
TENSOR_DIR="${BASE_DIR}/${EXPERIMENT_NAME}"
WEIGHTS_PATH="${BASE_DIR}/aix3-7b-base (1).pt"

BATCH_SIZE=16
NUM_BATCHES=100
NUM_GPUS=8

mkdir -p logs

echo "============================================================"
echo "Activating conda environment..."
echo "============================================================"
conda activate python3112

# GPU diagnostics
echo ""
echo "============================================================"
echo "GPU Information"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.free --format=csv
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Find k=1 dataset
DATA_DIR=$(find "${TENSOR_DIR}" -maxdepth 1 -type d -name "juliet_c_simhash_k=1_*_seed42" | head -n 1)
if [ -z "${DATA_DIR}" ]; then
    echo "ERROR: Could not find k=1 dataset"
    exit 1
fi
echo "Using dataset: ${DATA_DIR}"

echo ""
echo "============================================================"
echo "Running DDP Benchmark"
echo "============================================================"
echo "  GPUs: ${NUM_GPUS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo ""

# Run with torchrun
torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    "${BASE_DIR}/benchmark_ddp.py" \
    --data-dir "${DATA_DIR}" \
    --weights "${WEIGHTS_PATH}" \
    --batch-size ${BATCH_SIZE} \
    --num-batches ${NUM_BATCHES}

echo ""
echo "============================================================"
echo "Benchmark Complete"
echo "============================================================"
