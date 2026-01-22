#!/bin/bash -lT
#SBATCH -J dedup_ddp
#SBATCH --output=logs/experiment_ddp_%j.log
#SBATCH --error=logs/experiment_ddp_%j.err
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -n 1
#SBATCH -p gpu_7day
#SBATCH --gres=gpu:8
#SBATCH --time=168:00:00

# ============================================================================
# Deduplication Experiment with DDP: LSTM Training on SimHash Datasets
# ============================================================================
# Uses DistributedDataParallel for efficient multi-GPU training.
# ============================================================================

# Don't exit on error - we want to continue with other k values if one fails
set +e

# Configuration
EXPERIMENT_NAME="Experiment_1"
BASE_DIR="/scratch/aeg00011"
TENSOR_DIR="${BASE_DIR}/${EXPERIMENT_NAME}"
OUTPUT_DIR="${BASE_DIR}/experiments/${EXPERIMENT_NAME}_ddp"
WEIGHTS_PATH="${BASE_DIR}/aix3-7b-base (1).pt"
OOD_DATASET_PATTERN="devign_*_seed42"

# Training parameters
EPOCHS=50
PATIENCE=5
BATCH_SIZE=16  # Per-GPU batch size (effective = 16 * 8 = 128)
LEARNING_RATE=0.001
SEEDS="42 123 456 789 1024"

# K values to process
K_VALUES="1 2 3 4 5 6 7 8 9 10 11 12"

# Number of GPUs
NUM_GPUS=8

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/summary"
mkdir -p logs

# Activate conda environment
echo "============================================================"
echo "Activating conda environment..."
echo "============================================================"
conda activate python3112

# Clear GPU memory
echo ""
echo "============================================================"
echo "Clearing GPU Memory..."
echo "============================================================"
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read pid; do
    if [ -n "$pid" ]; then
        echo "Killing leftover GPU process: $pid"
        kill -9 $pid 2>/dev/null || true
    fi
done

python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    print('GPU memory cleared on all devices')
"

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
echo "DDP Deduplication Experiment Started"
echo "Time: $(date)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Experiment Name: ${EXPERIMENT_NAME}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  Number of GPUs: ${NUM_GPUS}"
echo "  Batch Size (per GPU): ${BATCH_SIZE}"
echo "  Effective Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "  Epochs: ${EPOCHS}"
echo "  Patience: ${PATIENCE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Seeds: ${SEEDS}"
echo "  K Values: ${K_VALUES}"
echo ""

# Find OOD dataset
echo "============================================================"
echo "Finding OOD dataset..."
echo "============================================================"
OOD_DIR=$(find "${TENSOR_DIR}" -maxdepth 1 -type d -name "${OOD_DATASET_PATTERN}" | head -n 1)

if [ -z "${OOD_DIR}" ]; then
    echo "ERROR: Could not find OOD dataset matching pattern '${OOD_DATASET_PATTERN}'"
    exit 1
fi

echo "Found OOD dataset: ${OOD_DIR}"

# List available datasets
echo ""
echo "============================================================"
echo "Available datasets:"
echo "============================================================"
ls -la "${TENSOR_DIR}"
echo ""

# Function to run single DDP experiment
run_ddp_experiment() {
    local k=$1
    local dataset_pattern="juliet_c_simhash_k=${k}_*_seed42"
    
    echo ""
    echo "============================================================"
    echo "Processing k=${k} with DDP (${NUM_GPUS} GPUs)"
    echo "============================================================"
    
    # Find dataset directory
    local dataset_dir=$(find "${TENSOR_DIR}" -maxdepth 1 -type d -name "${dataset_pattern}" | head -n 1)
    
    if [ -z "${dataset_dir}" ]; then
        echo "WARNING: Could not find dataset for k=${k}, skipping..."
        return 1
    fi
    
    local dataset_name="juliet_c_simhash_k=${k}"
    local log_file="${OUTPUT_DIR}/logs/${dataset_name}.log"
    
    echo "Dataset directory: ${dataset_dir}"
    echo "Output will be logged to: ${log_file}"
    echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
    
    # Run training with torchrun for DDP
    # --standalone: single-node training
    # --nproc_per_node: number of processes (GPUs) per node
    torchrun \
        --standalone \
        --nproc_per_node=${NUM_GPUS} \
        "${BASE_DIR}/train_lstm_ddp.py" \
        --dataset-dir "${dataset_dir}" \
        --dataset-name "${dataset_name}" \
        --ood-dir "${OOD_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --weights "${WEIGHTS_PATH}" \
        --batch-size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --patience ${PATIENCE} \
        --lr ${LEARNING_RATE} \
        --seeds ${SEEDS} \
        2>&1 | tee "${log_file}"
    
    local status=$?
    
    if [ $status -eq 0 ]; then
        echo "✓ Completed k=${k} successfully"
    else
        echo "✗ Failed k=${k} with status ${status}"
    fi
    
    # Clear GPU memory between experiments
    python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    print('GPU memory cleared')
"
    
    return $status
}

# Run experiments
echo "============================================================"
echo "Starting DDP experiments..."
echo "============================================================"

successful=0
failed=0

for k in ${K_VALUES}; do
    if run_ddp_experiment $k; then
        successful=$((successful + 1))
    else
        failed=$((failed + 1))
    fi
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_FORMATTED=$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

# Generate summary
echo ""
echo "============================================================"
echo "Generating Summary..."
echo "============================================================"

SUMMARY_FILE="${OUTPUT_DIR}/summary/experiment_summary.txt"

cat > "${SUMMARY_FILE}" << EOF
============================================================
DDP DEDUPLICATION EXPERIMENT SUMMARY
============================================================
Experiment: ${EXPERIMENT_NAME}
Date: $(date)
Total Runtime: ${ELAPSED_FORMATTED}

Configuration:
  Number of GPUs: ${NUM_GPUS}
  Batch Size (per GPU): ${BATCH_SIZE}
  Effective Batch Size: $((BATCH_SIZE * NUM_GPUS))
  Epochs: ${EPOCHS}
  Patience: ${PATIENCE}
  Learning Rate: ${LEARNING_RATE}
  Seeds: ${SEEDS}

Results:
  Successful: ${successful}/12
  Failed: ${failed}/12

OOD Dataset: ${OOD_DIR}

Individual Results:
EOF

# Append individual results
for k in ${K_VALUES}; do
    results_file="${OUTPUT_DIR}/juliet_c_simhash_k=${k}/results/experiment_results.json"
    if [ -f "${results_file}" ]; then
        echo "" >> "${SUMMARY_FILE}"
        echo "--- k=${k} ---" >> "${SUMMARY_FILE}"
        python -c "
import json
with open('${results_file}') as f:
    data = json.load(f)
agg = data.get('aggregate', {})
print(f\"  Test AUROC: {agg.get('test_auroc_mean', 'N/A'):.4f} ± {agg.get('test_auroc_std', 'N/A'):.4f}\")
print(f\"  Test F1: {agg.get('test_f1_mean', 'N/A'):.4f} ± {agg.get('test_f1_std', 'N/A'):.4f}\")
print(f\"  OOD AUROC: {agg.get('ood_auroc_mean', 'N/A'):.4f} ± {agg.get('ood_auroc_std', 'N/A'):.4f}\")
print(f\"  OOD F1: {agg.get('ood_f1_mean', 'N/A'):.4f} ± {agg.get('ood_f1_std', 'N/A'):.4f}\")
" >> "${SUMMARY_FILE}" 2>/dev/null || echo "  Results not available" >> "${SUMMARY_FILE}"
    else
        echo "" >> "${SUMMARY_FILE}"
        echo "--- k=${k} ---" >> "${SUMMARY_FILE}"
        echo "  Results not available" >> "${SUMMARY_FILE}"
    fi
done

# Create CSV summary
CSV_FILE="${OUTPUT_DIR}/summary/results_summary.csv"
echo "k,test_auroc_mean,test_auroc_std,test_f1_mean,test_f1_std,ood_auroc_mean,ood_auroc_std,ood_f1_mean,ood_f1_std" > "${CSV_FILE}"

for k in ${K_VALUES}; do
    results_file="${OUTPUT_DIR}/juliet_c_simhash_k=${k}/results/experiment_results.json"
    if [ -f "${results_file}" ]; then
        python -c "
import json
with open('${results_file}') as f:
    data = json.load(f)
agg = data.get('aggregate', {})
print(f\"${k},{agg.get('test_auroc_mean', '')},{agg.get('test_auroc_std', '')},{agg.get('test_f1_mean', '')},{agg.get('test_f1_std', '')},{agg.get('ood_auroc_mean', '')},{agg.get('ood_auroc_std', '')},{agg.get('ood_f1_mean', '')},{agg.get('ood_f1_std', '')}\")
" >> "${CSV_FILE}" 2>/dev/null || echo "${k},,,,,,,," >> "${CSV_FILE}"
    else
        echo "${k},,,,,,,," >> "${CSV_FILE}"
    fi
done

echo "Summary saved to: ${SUMMARY_FILE}"
echo "CSV results saved to: ${CSV_FILE}"

# Print final summary
echo ""
echo "============================================================"
echo "DDP EXPERIMENT COMPLETE"
echo "============================================================"
echo ""
cat "${SUMMARY_FILE}"
echo ""
echo "============================================================"
echo "Output files:"
echo "  Summary: ${SUMMARY_FILE}"
echo "  CSV: ${CSV_FILE}"
echo "  Logs: ${OUTPUT_DIR}/logs/"
echo "  Models: ${OUTPUT_DIR}/juliet_c_simhash_k=*/results/"
echo "============================================================"
