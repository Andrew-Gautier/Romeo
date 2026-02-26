#!/bin/bash -lT
#SBATCH -J lora_vuln
#SBATCH --output=logs/lora_experiment_%j.log
#SBATCH --error=logs/lora_experiment_%j.err
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -n 1
#SBATCH -p gpu_7day
#SBATCH --gres=gpu:8
#SBATCH --time=168:00:00
#SBATCH --mem=256G

# ============================================================================
# LoRA Fine-Tuning Experiment: Transformer + LoRA for Vulnerability Detection
# ============================================================================
#
# This experiment fine-tunes full pretrained transformers with LoRA adapters
# for binary vulnerability classification. This is fundamentally different 
# from the LSTM embedding experiments (Experiments 1-4).
#
# IMPORTANT: The tensors MUST be tokenized with the model-specific tokenizer!
#   - DeepSeek tensors: tokenized with deepseek-ai/deepseek-coder-6.7b-base
#   - CodeLlama tensors: tokenized with codellama/CodeLlama-7b-hf
#
# Models:
#   1. DeepSeek-Coder-6.7B + LoRA
#   2. CodeLlama-7B + LoRA
#
# Dataset: Devign (train/val/test) with Juliet C as OOD
# ============================================================================

set +e

# Configuration
BASE_DIR="/scratch/aeg00011"
OUTPUT_DIR="${BASE_DIR}/experiments/LoRA_Experiments"

# HuggingFace cache
HF_CACHE_DIR="${BASE_DIR}/huggingface"

# ============================================================================
# TENSOR DIRECTORIES
# ============================================================================
# These must point to tensors tokenized with the MATCHING model tokenizer.
# If you used the Dataset Creation notebook with MODEL_CHOICE="deepseek",
# set the path for DEEPSEEK_TENSOR_DIR to that experiment's output.
#
# Example directory structure:
#   /scratch/aeg00011/Experiment_LoRA_DeepSeek/devign_YYYYMMDD_seed42/
#     train_sequences.pt, train_labels.pt, val_sequences.pt, etc.
# ============================================================================

# DeepSeek experiment tensors (tokenized with DeepSeek tokenizer)
DEEPSEEK_EXPERIMENT_DIR="${BASE_DIR}/Experiment_LoRA_DeepSeek"
DEEPSEEK_DATASET_DIR=$(find "${DEEPSEEK_EXPERIMENT_DIR}" -maxdepth 1 -type d -name "devign_*_seed42" 2>/dev/null | head -n 1)
DEEPSEEK_OOD_DIR=$(find "${DEEPSEEK_EXPERIMENT_DIR}" -maxdepth 1 -type d -name "devign_ood_*_seed42" 2>/dev/null | head -n 1)

# CodeLlama experiment tensors (tokenized with CodeLlama tokenizer)  
CODELLAMA_EXPERIMENT_DIR="${BASE_DIR}/Experiment_LoRA_CodeLlama"
CODELLAMA_DATASET_DIR=$(find "${CODELLAMA_EXPERIMENT_DIR}" -maxdepth 1 -type d -name "devign_*_seed42" 2>/dev/null | head -n 1)
CODELLAMA_OOD_DIR=$(find "${CODELLAMA_EXPERIMENT_DIR}" -maxdepth 1 -type d -name "devign_ood_*_seed42" 2>/dev/null | head -n 1)

# Training parameters
EPOCHS=10
PATIENCE=3
BATCH_SIZE=4            # Per-GPU (LoRA models are large — 6-7B params)
GRAD_ACCUM=8            # Effective batch = 4 * 8 = 32
LEARNING_RATE=2e-4
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

# LoRA hyperparameters
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# Activate conda environment
echo "============================================================"
echo "Activating conda environment..."
echo "============================================================"
conda activate python3112hf

# Install PEFT if not already installed
pip install peft accelerate bitsandbytes 2>/dev/null || true

# GPU diagnostics
echo ""
echo "============================================================"
echo "GPU Diagnostic Information"
echo "============================================================"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
python -c "
import torch
print(f'PyTorch CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
try:
    from peft import __version__ as peft_ver
    print(f'PEFT version: {peft_ver}')
except ImportError:
    print('PEFT NOT installed!')
"
echo "============================================================"

# Log start time
START_TIME=$(date +%s)
echo ""
echo "============================================================"
echo "LoRA Fine-Tuning Experiment Started"
echo "Time: $(date)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  HuggingFace Cache: ${HF_CACHE_DIR}"
echo "  Epochs: ${EPOCHS}"
echo "  Patience: ${PATIENCE}"
echo "  Batch Size (per GPU): ${BATCH_SIZE}"
echo "  Gradient Accumulation: ${GRAD_ACCUM}"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  LoRA Rank: ${LORA_R}"
echo "  LoRA Alpha: ${LORA_ALPHA}"
echo "  Seeds: ${SEEDS}"
echo ""

# ============================================================================
# Experiment 1: DeepSeek-Coder + LoRA
# ============================================================================
run_deepseek_lora() {
    echo ""
    echo "############################################################"
    echo "# DeepSeek-Coder-6.7B + LoRA"
    echo "############################################################"
    
    if [ -z "${DEEPSEEK_DATASET_DIR}" ]; then
        echo "ERROR: DeepSeek tensor directory not found in ${DEEPSEEK_EXPERIMENT_DIR}"
        echo "  Did you run the Dataset Creation notebook with MODEL_CHOICE='deepseek'?"
        return 1
    fi
    
    if [ -z "${DEEPSEEK_OOD_DIR}" ]; then
        echo "WARNING: DeepSeek OOD directory not found, using Juliet k=1 as fallback"
        DEEPSEEK_OOD_DIR=$(find "${DEEPSEEK_EXPERIMENT_DIR}" -maxdepth 1 -type d -name "juliet_*k=1*_seed42" 2>/dev/null | head -n 1)
        if [ -z "${DEEPSEEK_OOD_DIR}" ]; then
            echo "ERROR: No OOD dataset found"
            return 1
        fi
    fi
    
    echo "Dataset: ${DEEPSEEK_DATASET_DIR}"
    echo "OOD: ${DEEPSEEK_OOD_DIR}"
    
    python "${BASE_DIR}/train_lora.py" \
        --model-name deepseek-coder \
        --dataset-dir "${DEEPSEEK_DATASET_DIR}" \
        --ood-dir "${DEEPSEEK_OOD_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --hf-cache-dir "${HF_CACHE_DIR}" \
        --batch-size ${BATCH_SIZE} \
        --grad-accum ${GRAD_ACCUM} \
        --lr ${LEARNING_RATE} \
        --epochs ${EPOCHS} \
        --patience ${PATIENCE} \
        --lora-r ${LORA_R} \
        --lora-alpha ${LORA_ALPHA} \
        --lora-dropout ${LORA_DROPOUT} \
        --seeds ${SEEDS}
    
    local status=$?
    echo ""
    if [ $status -eq 0 ]; then
        echo "✓ DeepSeek-Coder LoRA experiment completed successfully"
    else
        echo "✗ DeepSeek-Coder LoRA experiment failed (status: $status)"
    fi
    
    # Clear GPU memory
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

# ============================================================================
# Experiment 2: CodeLlama + LoRA
# ============================================================================
run_codellama_lora() {
    echo ""
    echo "############################################################"
    echo "# CodeLlama-7B + LoRA"
    echo "############################################################"
    
    if [ -z "${CODELLAMA_DATASET_DIR}" ]; then
        echo "ERROR: CodeLlama tensor directory not found in ${CODELLAMA_EXPERIMENT_DIR}"
        echo "  Did you run the Dataset Creation notebook with MODEL_CHOICE='codellama'?"
        return 1
    fi
    
    if [ -z "${CODELLAMA_OOD_DIR}" ]; then
        echo "WARNING: CodeLlama OOD directory not found, using Juliet k=1 as fallback"
        CODELLAMA_OOD_DIR=$(find "${CODELLAMA_EXPERIMENT_DIR}" -maxdepth 1 -type d -name "juliet_*k=1*_seed42" 2>/dev/null | head -n 1)
        if [ -z "${CODELLAMA_OOD_DIR}" ]; then
            echo "ERROR: No OOD dataset found"
            return 1
        fi
    fi
    
    echo "Dataset: ${CODELLAMA_DATASET_DIR}"
    echo "OOD: ${CODELLAMA_OOD_DIR}"
    
    python "${BASE_DIR}/train_lora.py" \
        --model-name codellama \
        --dataset-dir "${CODELLAMA_DATASET_DIR}" \
        --ood-dir "${CODELLAMA_OOD_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --hf-cache-dir "${HF_CACHE_DIR}" \
        --batch-size ${BATCH_SIZE} \
        --grad-accum ${GRAD_ACCUM} \
        --lr ${LEARNING_RATE} \
        --epochs ${EPOCHS} \
        --patience ${PATIENCE} \
        --lora-r ${LORA_R} \
        --lora-alpha ${LORA_ALPHA} \
        --lora-dropout ${LORA_DROPOUT} \
        --seeds ${SEEDS}
    
    local status=$?
    echo ""
    if [ $status -eq 0 ]; then
        echo "✓ CodeLlama LoRA experiment completed successfully"
    else
        echo "✗ CodeLlama LoRA experiment failed (status: $status)"
    fi
    
    # Clear GPU memory
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

# ============================================================================
# Run experiments
# ============================================================================

echo "============================================================"
echo "Starting LoRA experiments..."
echo "============================================================"

deepseek_status=0
codellama_status=0

# Run DeepSeek
run_deepseek_lora
deepseek_status=$?

# Run CodeLlama
run_codellama_lora
codellama_status=$?

# ============================================================================
# Final Summary
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_FORMATTED=$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

echo ""
echo "============================================================"
echo "LoRA EXPERIMENT COMPLETE"
echo "============================================================"
echo "Total Runtime: ${ELAPSED_FORMATTED}"
echo ""
echo "Results:"
echo "  DeepSeek-Coder: $([ $deepseek_status -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "  CodeLlama:      $([ $codellama_status -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo ""
echo "Output files:"
echo "  ${OUTPUT_DIR}/lora_deepseek-coder/results/"
echo "  ${OUTPUT_DIR}/lora_codellama/results/"
echo "============================================================"

# Print result summaries if available
for model in "deepseek-coder" "codellama"; do
    summary="${OUTPUT_DIR}/lora_${model}/results/results_summary.csv"
    if [ -f "${summary}" ]; then
        echo ""
        echo "--- ${model} ---"
        cat "${summary}"
    fi
done
