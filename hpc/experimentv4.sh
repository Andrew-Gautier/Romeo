#!/bin/bash -lT
#SBATCH -J exp4_deepseek
#SBATCH --output=logs/experiment4_ddp_%j.log
#SBATCH --error=logs/experiment4_ddp_%j.err
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
# Experiment 4: DeepSeek Coder embeddings from HuggingFace, Train/test/split on Devign dataset, OOD is Juliet_C
# Enhanced with GPU monitoring and detailed per-seed metrics
# ============================================================================

# Don't exit on error - we want to continue with other k values if one fails
set +e

# Configuration
EXPERIMENT_NAME="Experiment_4"
BASE_DIR="/scratch/aeg00011"
TENSOR_DIR="${BASE_DIR}/${EXPERIMENT_NAME}"
OUTPUT_DIR="${BASE_DIR}/experiments/${EXPERIMENT_NAME}"

# HuggingFace model configuration
HF_CACHE_DIR="${BASE_DIR}/huggingface"
MODEL_NAME="deepseek-coder"  # Options: deepseek-coder, codellama

OOD_DATASET_PATTERN="juliet_*_seed42"

# Training parameters
EPOCHS=50
PATIENCE=5
BATCH_SIZE=16  # Per-GPU batch size (effective = 16 * 8 = 128)
LEARNING_RATE=0.001
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

# K values to process
K_VALUES="1 2 3 4 5 6 7 8 9 10 11 12"

# Number of GPUs
NUM_GPUS=8

# GPU monitoring parameters
GPU_MONITOR_INTERVAL=30  # Sample GPU stats every 30 seconds
GPU_LOG_DIR="${OUTPUT_DIR}/gpu_logs"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/summary"
mkdir -p "${GPU_LOG_DIR}"
mkdir -p logs

# Activate conda environment
echo "============================================================"
echo "Activating conda environment..."
echo "============================================================"
conda activate python3112hf

# Verify torchrun is available
echo "Python path: $(which python)"
echo "Torchrun path: $(which torchrun || echo 'NOT FOUND')"

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
echo "  HuggingFace Cache: ${HF_CACHE_DIR}"
echo "  Model Name: ${MODEL_NAME}"
echo "  Number of GPUs: ${NUM_GPUS}"
echo "  Batch Size (per GPU): ${BATCH_SIZE}"
echo "  Effective Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "  Epochs: ${EPOCHS}"
echo "  Patience: ${PATIENCE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Seeds: ${SEEDS}"
echo "  K Values: ${K_VALUES}"
echo "  GPU Monitoring Interval: ${GPU_MONITOR_INTERVAL}s"
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

# Function to start GPU monitoring
start_gpu_monitor() {
    local k=$1
    local log_file="${GPU_LOG_DIR}/gpu_stats_k${k}.csv"
    local pid_file="${GPU_LOG_DIR}/monitor_k${k}.pid"
    
    # Create CSV header
    echo "timestamp,gpu_id,utilization_gpu,utilization_memory,memory_used_mb,memory_total_mb,memory_free_mb,temperature,power_draw_w,power_limit_w" > "${log_file}"
    
    # Start background monitoring process
    (
        while true; do
            timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,memory.free,temperature.gpu,power.draw,power.limit \
                --format=csv,noheader,nounits | while IFS=, read -r gpu_id util_gpu util_mem mem_used mem_total mem_free temp power_draw power_limit; do
                echo "${timestamp},${gpu_id},${util_gpu},${util_mem},${mem_used},${mem_total},${mem_free},${temp},${power_draw},${power_limit}" >> "${log_file}"
            done
            sleep ${GPU_MONITOR_INTERVAL}
        done
    ) &
    
    echo $! > "${pid_file}"
    echo "GPU monitoring started (PID: $!, interval: ${GPU_MONITOR_INTERVAL}s, log: ${log_file})"
}

# Function to stop GPU monitoring
stop_gpu_monitor() {
    local k=$1
    local pid_file="${GPU_LOG_DIR}/monitor_k${k}.pid"
    
    if [ -f "${pid_file}" ]; then
        local pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null
            echo "GPU monitoring stopped (PID: ${pid})"
        fi
        rm -f "${pid_file}"
    fi
}

# Function to generate GPU statistics summary
generate_gpu_stats() {
    local k=$1
    local log_file="${GPU_LOG_DIR}/gpu_stats_k${k}.csv"
    local summary_file="${GPU_LOG_DIR}/gpu_summary_k${k}.txt"
    
    if [ ! -f "${log_file}" ]; then
        echo "No GPU stats found for k=${k}"
        return
    fi
    
    python - << EOF > "${summary_file}"
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('${log_file}')
    
    if len(df) == 0:
        print("No GPU statistics collected for k=${k}")
    else:
        print("=" * 60)
        print(f"GPU Statistics Summary for k=${k}")
        print("=" * 60)
        print(f"Total samples: {len(df)}")
        print(f"Duration: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        print()
        
        for gpu_id in sorted(df['gpu_id'].unique()):
            gpu_df = df[df['gpu_id'] == gpu_id]
            print(f"GPU {gpu_id}:")
            print(f"  Utilization (GPU):    {gpu_df['utilization_gpu'].mean():.1f}% (mean), {gpu_df['utilization_gpu'].max():.1f}% (max)")
            print(f"  Utilization (Memory): {gpu_df['utilization_memory'].mean():.1f}% (mean), {gpu_df['utilization_memory'].max():.1f}% (max)")
            print(f"  Memory Used:          {gpu_df['memory_used_mb'].mean():.0f} MB (mean), {gpu_df['memory_used_mb'].max():.0f} MB (max)")
            print(f"  Memory Total:         {gpu_df['memory_total_mb'].iloc[0]:.0f} MB")
            print(f"  Temperature:          {gpu_df['temperature'].mean():.1f}°C (mean), {gpu_df['temperature'].max():.1f}°C (max)")
            print(f"  Power Draw:           {gpu_df['power_draw_w'].mean():.1f}W (mean), {gpu_df['power_draw_w'].max():.1f}W (max)")
            print()
        
        # Overall statistics
        print("Overall Statistics:")
        print(f"  Average GPU Utilization: {df['utilization_gpu'].mean():.1f}%")
        print(f"  Average Memory Utilization: {df['utilization_memory'].mean():.1f}%")
        print(f"  Average Power Draw: {df['power_draw_w'].mean():.1f}W")
        print(f"  Peak Memory Used: {df['memory_used_mb'].max():.0f} MB (GPU {df.loc[df['memory_used_mb'].idxmax(), 'gpu_id']})")
        print(f"  Peak Temperature: {df['temperature'].max():.1f}°C (GPU {df.loc[df['temperature'].idxmax(), 'gpu_id']})")
        print("=" * 60)
        
except Exception as e:
    print(f"Error generating GPU statistics: {e}")
EOF
    
    cat "${summary_file}"
}

# Function to extract per-seed metrics from experiment results
extract_per_seed_metrics() {
    local k=$1
    local results_file="${OUTPUT_DIR}/devign_simhash_k=${k}/results/experiment_results.json"
    local output_csv="${OUTPUT_DIR}/summary/per_seed_metrics_k${k}.csv"
    
    if [ ! -f "${results_file}" ]; then
        echo "Results file not found for k=${k}: ${results_file}"
        return
    fi
    
    python - << EOF
import json
import csv

try:
    with open('${results_file}', 'r') as f:
        data = json.load(f)
    
    # Extract per-seed results
    seeds_data = data.get('seeds', {})
    
    if not seeds_data:
        print("No per-seed data found for k=${k}")
    else:
        # Prepare CSV
        with open('${output_csv}', 'w', newline='') as csvfile:
            fieldnames = ['k', 'seed', 'split', 'auroc', 'f1', 'accuracy', 'precision', 'recall', 'loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for seed, seed_results in sorted(seeds_data.items()):
                seed_num = seed.replace('seed_', '')
                
                # Test metrics
                test_metrics = seed_results.get('test', {})
                writer.writerow({
                    'k': ${k},
                    'seed': seed_num,
                    'split': 'test',
                    'auroc': f"{test_metrics.get('auroc', ''):.6f}" if test_metrics.get('auroc') else '',
                    'f1': f"{test_metrics.get('f1', ''):.6f}" if test_metrics.get('f1') else '',
                    'accuracy': f"{test_metrics.get('accuracy', ''):.6f}" if test_metrics.get('accuracy') else '',
                    'precision': f"{test_metrics.get('precision', ''):.6f}" if test_metrics.get('precision') else '',
                    'recall': f"{test_metrics.get('recall', ''):.6f}" if test_metrics.get('recall') else '',
                    'loss': f"{test_metrics.get('loss', ''):.6f}" if test_metrics.get('loss') else '',
                })
                
                # OOD metrics
                ood_metrics = seed_results.get('ood', {})
                writer.writerow({
                    'k': ${k},
                    'seed': seed_num,
                    'split': 'ood',
                    'auroc': f"{ood_metrics.get('auroc', ''):.6f}" if ood_metrics.get('auroc') else '',
                    'f1': f"{ood_metrics.get('f1', ''):.6f}" if ood_metrics.get('f1') else '',
                    'accuracy': f"{ood_metrics.get('accuracy', ''):.6f}" if ood_metrics.get('accuracy') else '',
                    'precision': f"{ood_metrics.get('precision', ''):.6f}" if ood_metrics.get('precision') else '',
                    'recall': f"{ood_metrics.get('recall', ''):.6f}" if ood_metrics.get('recall') else '',
                    'loss': f"{ood_metrics.get('loss', ''):.6f}" if ood_metrics.get('loss') else '',
                })
        
        print(f"Per-seed metrics exported to: ${output_csv}")
        
except Exception as e:
    print(f"Error extracting per-seed metrics for k=${k}: {e}")
    import traceback
    traceback.print_exc()
EOF
}

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
    
    local dataset_name="devign_simhash_k=${k}"
    local log_file="${OUTPUT_DIR}/logs/${dataset_name}.log"
    
    echo "Dataset directory: ${dataset_dir}"
    echo "Output will be logged to: ${log_file}"
    echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
    
    # Start GPU monitoring
    start_gpu_monitor ${k}
    
    # Capture GPU state before training
    echo ""
    echo "--- GPU State Before Training (k=${k}) ---"
    nvidia-smi
    echo "-------------------------------------------"
    
    # Run training with torchrun for DDP
    # --standalone: single-node training
    # --nproc_per_node: number of processes (GPUs) per node
    torchrun \
        --standalone \
        --nproc_per_node=${NUM_GPUS} \
        "${BASE_DIR}/train_lstm_ddp2.py" \
        --dataset-dir "${dataset_dir}" \
        --dataset-name "${dataset_name}" \
        --ood-dir "${OOD_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --hf-cache-dir "${HF_CACHE_DIR}" \
        --model-name "${MODEL_NAME}" \
        --batch-size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --patience ${PATIENCE} \
        --lr ${LEARNING_RATE} \
        --seeds ${SEEDS} \
        2>&1 | tee "${log_file}"
    
    local status=$?
    
    # Stop GPU monitoring
    stop_gpu_monitor ${k}
    
    # Capture GPU state after training
    echo ""
    echo "--- GPU State After Training (k=${k}) ---"
    nvidia-smi
    echo "-------------------------------------------"
    
    # Generate GPU statistics summary
    echo ""
    generate_gpu_stats ${k}
    
    # Extract per-seed metrics
    echo ""
    echo "Extracting per-seed metrics for k=${k}..."
    extract_per_seed_metrics ${k}
    
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
  HuggingFace Cache: ${HF_CACHE_DIR}
  Model Name: ${MODEL_NAME}
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
    results_file="${OUTPUT_DIR}/devign_simhash_k=${k}/results/experiment_results.json"
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

# Create aggregate CSV summary
CSV_FILE="${OUTPUT_DIR}/summary/results_summary.csv"
echo "k,test_auroc_mean,test_auroc_std,test_f1_mean,test_f1_std,ood_auroc_mean,ood_auroc_std,ood_f1_mean,ood_f1_std" > "${CSV_FILE}"

for k in ${K_VALUES}; do
    results_file="${OUTPUT_DIR}/devign_simhash_k=${k}/results/experiment_results.json"
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

# Consolidate all per-seed metrics into a single master file
echo ""
echo "============================================================"
echo "Consolidating all per-seed metrics..."
echo "============================================================"

MASTER_SEED_CSV="${OUTPUT_DIR}/summary/all_seeds_metrics.csv"
echo "k,seed,split,auroc,f1,accuracy,precision,recall,loss" > "${MASTER_SEED_CSV}"

for k in ${K_VALUES}; do
    per_seed_file="${OUTPUT_DIR}/summary/per_seed_metrics_k${k}.csv"
    if [ -f "${per_seed_file}" ]; then
        # Skip header and append to master file
        tail -n +2 "${per_seed_file}" >> "${MASTER_SEED_CSV}"
    fi
done

if [ -f "${MASTER_SEED_CSV}" ]; then
    num_rows=$(tail -n +2 "${MASTER_SEED_CSV}" | wc -l)
    echo "Consolidated per-seed metrics: ${num_rows} rows"
    echo "Master file: ${MASTER_SEED_CSV}"
fi

# Generate consolidated GPU statistics
echo ""
echo "============================================================"
echo "Consolidating GPU Statistics..."
echo "============================================================"

GPU_SUMMARY_FILE="${OUTPUT_DIR}/summary/gpu_utilization_summary.txt"

cat > "${GPU_SUMMARY_FILE}" << EOF
============================================================
GPU UTILIZATION SUMMARY - ALL EXPERIMENTS
============================================================
Date: $(date)
Monitoring Interval: ${GPU_MONITOR_INTERVAL}s

EOF

for k in ${K_VALUES}; do
    gpu_summary="${GPU_LOG_DIR}/gpu_summary_k${k}.txt"
    if [ -f "${gpu_summary}" ]; then
        cat "${gpu_summary}" >> "${GPU_SUMMARY_FILE}"
        echo "" >> "${GPU_SUMMARY_FILE}"
    fi
done

echo "Summary saved to: ${SUMMARY_FILE}"
echo "CSV results saved to: ${CSV_FILE}"
echo "Master per-seed metrics: ${MASTER_SEED_CSV}"
echo "GPU statistics saved to: ${GPU_SUMMARY_FILE}"
echo "Individual GPU logs: ${GPU_LOG_DIR}/"

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
echo "  Aggregate CSV: ${CSV_FILE}"
echo "  Per-Seed Metrics (Master): ${MASTER_SEED_CSV}"
echo "  Per-Seed Metrics (Individual): ${OUTPUT_DIR}/summary/per_seed_metrics_k*.csv"
echo "  GPU Stats: ${GPU_SUMMARY_FILE}"
echo "  GPU Logs (Raw CSV): ${GPU_LOG_DIR}/"
echo "  Training Logs: ${OUTPUT_DIR}/logs/"
echo "  Models: ${OUTPUT_DIR}/devign_simhash_k=*/results/"
echo "============================================================"
