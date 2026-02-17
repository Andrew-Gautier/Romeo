#!/bin/bash -lT
#SBATCH -J eval_no_ft
#SBATCH --output=logs/eval_no_finetune_%j.log
#SBATCH --error=logs/eval_no_finetune_%j.err
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -n 1
#SBATCH -p gpu_7day
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

# ============================================================================
# Embedding Evaluation (No Fine-Tuning)
# ============================================================================
# Evaluates raw pretrained embeddings from 3 models on Devign test set
# using the LSTM+attention classifier with NO training.
#
# Only 1 GPU needed — this is inference-only, no DDP required.
# ============================================================================

set +e

# Configuration
BASE_DIR="/scratch/aeg00011"
OUTPUT_DIR="${BASE_DIR}/experiments/No_Finetune_Eval"

# Embedding sources
AIXCODER_WEIGHTS="${BASE_DIR}/aix3-7b-base (1).pt"
HF_CACHE_DIR="${BASE_DIR}/huggingface"

# Devign test data — use the k=1 (no deduplication) version for full dataset
# This is the same tensor directory used in Experiment 4
DEVIGN_DIR="${BASE_DIR}/Experiment_4"

# Find the Devign tensor directory (devign_*_seed42)
DEVIGN_TENSOR_DIR=$(find "${DEVIGN_DIR}" -maxdepth 1 -type d -name "devign_*_seed42" 2>/dev/null | head -n 1)

# If not found in Experiment_4, try looking for it as juliet tensors from Experiment 1
if [ -z "${DEVIGN_TENSOR_DIR}" ]; then
    # Experiment 1 uses devign as OOD — look for devign tensors there
    DEVIGN_TENSOR_DIR=$(find "${BASE_DIR}/Experiment_1" -maxdepth 1 -type d -name "devign_*_seed42" 2>/dev/null | head -n 1)
fi

if [ -z "${DEVIGN_TENSOR_DIR}" ]; then
    echo "ERROR: Could not find Devign tensor directory"
    echo "Searched in: ${DEVIGN_DIR} and ${BASE_DIR}/Experiment_1"
    exit 1
fi

# Training parameters
BATCH_SIZE=32
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# Activate conda environment (same as experiments 2/3/4)
echo "============================================================"
echo "Activating conda environment..."
echo "============================================================"
conda activate python3112hf

# GPU diagnostics
echo ""
echo "============================================================"
echo "GPU Diagnostic Information"
echo "============================================================"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo "============================================================"

# Log start time
START_TIME=$(date +%s)
echo ""
echo "============================================================"
echo "Embedding Evaluation (No Fine-Tuning) Started"
echo "Time: $(date)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  aiXcoder Weights: ${AIXCODER_WEIGHTS}"
echo "  HuggingFace Cache: ${HF_CACHE_DIR}"
echo "  Devign Tensors: ${DEVIGN_TENSOR_DIR}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Seeds: ${SEEDS}"
echo ""

# Verify files exist
echo "============================================================"
echo "Verifying data paths..."
echo "============================================================"

if [ -f "${AIXCODER_WEIGHTS}" ]; then
    echo "✓ aiXcoder weights found: ${AIXCODER_WEIGHTS}"
else
    echo "✗ aiXcoder weights NOT found: ${AIXCODER_WEIGHTS}"
fi

if [ -d "${HF_CACHE_DIR}" ]; then
    echo "✓ HuggingFace cache found: ${HF_CACHE_DIR}"
    ls -d "${HF_CACHE_DIR}"/models--* 2>/dev/null | while read d; do
        echo "    $(basename $d)"
    done
else
    echo "✗ HuggingFace cache NOT found: ${HF_CACHE_DIR}"
fi

if [ -f "${DEVIGN_TENSOR_DIR}/test_sequences.pt" ]; then
    echo "✓ Devign test data found: ${DEVIGN_TENSOR_DIR}"
    ls -lh "${DEVIGN_TENSOR_DIR}"/test_*.pt
else
    echo "✗ Devign test_sequences.pt NOT found in: ${DEVIGN_TENSOR_DIR}"
    echo "  Available files:"
    ls -la "${DEVIGN_TENSOR_DIR}"
fi

echo ""

# Run evaluation
echo "============================================================"
echo "Running evaluation..."
echo "============================================================"

python "${BASE_DIR}/evaluate_embeddings_no_finetune.py" \
    --devign-dir "${DEVIGN_TENSOR_DIR}" \
    --aixcoder-weights "${AIXCODER_WEIGHTS}" \
    --hf-cache-dir "${HF_CACHE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size ${BATCH_SIZE} \
    --seeds ${SEEDS} \
    --models aixcoder deepseek-coder codellama

STATUS=$?

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_FORMATTED=$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

echo ""
echo "============================================================"
echo "EVALUATION COMPLETE"
echo "============================================================"
echo "Status: $([ $STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "Total Runtime: ${ELAPSED_FORMATTED}"
echo ""
echo "Output files:"
echo "  ${OUTPUT_DIR}/no_finetune_results.json"
echo "  ${OUTPUT_DIR}/no_finetune_summary.csv"
echo "  ${OUTPUT_DIR}/no_finetune_per_seed.csv"
echo "  ${OUTPUT_DIR}/no_finetune_summary.txt"
echo "  ${OUTPUT_DIR}/no_finetune_comparison.png"
echo "  ${OUTPUT_DIR}/no_finetune_comparison.pdf"
echo "  ${OUTPUT_DIR}/no_finetune_boxplot.png"
echo "  ${OUTPUT_DIR}/no_finetune_boxplot.pdf"
echo "============================================================"

# Print summary if available
if [ -f "${OUTPUT_DIR}/no_finetune_summary.txt" ]; then
    echo ""
    cat "${OUTPUT_DIR}/no_finetune_summary.txt"
fi
