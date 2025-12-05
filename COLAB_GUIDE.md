# Google Colab Training Workflow

## Quick Start Guide

### 1. Setup Google Colab
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install transformers torchmetrics tqdm

# Clone or upload your code
!git clone https://github.com/Andrew-Gautier/Romeo.git
# OR upload load_and_predict.py manually
```

### 2. Prepare Data Structure
Upload your tensor files to Google Drive in this structure:
```
/content/drive/MyDrive/romeo/
├── pretraining/
│   ├── c/
│   │   ├── train/
│   │   │   ├── sequences.pt
│   │   │   └── labels.pt
│   │   └── validation/
│   │       ├── sequences.pt
│   │       └── labels.pt
│   ├── java/
│   │   ├── train/
│   │   └── validation/
│   ├── csharp/
│   │   ├── train/
│   │   └── validation/
│   └── combined/
│       ├── train/
│       └── validation/
└── models/  (will be created automatically)
```

### 3. Run Training
```python
# Navigate to code directory
%cd /content/Romeo

# Run the training script
!python load_and_predict.py
```

This will:
- Download embeddings from HuggingFace (aiXcoder/aixcoder-7b-base)
- Train 4 models: C, Java, C#, and Combined
- Save models to `/content/drive/MyDrive/romeo/models/`
- Generate training curve plots
- Show progress bars and metrics

### 4. Monitor Training
Expected output:
```
================================================================================
LSTM Vulnerability Detection - Pretraining on Multiple Languages
================================================================================

================================================================================
Training model for: C
================================================================================
Loading data from /content/drive/MyDrive/romeo/pretraining/c...
  Loaded train - Sequences: torch.Size([1400, 4096]), Labels: torch.Size([1400])
  Positive samples: 700/1400 (50.00%)
  Loaded validation - Sequences: torch.Size([600, 4096]), Labels: torch.Size([600])
  Positive samples: 300/600 (50.00%)
Train batches: 43, Val batches: 19
Model parameters: 1,075,331,073

Training started at: 2024-12-05 15:30:00
Training: 100%|██████████| 43/43 [01:23<00:00,  1.93s/it]
Evaluating: 100%|██████████| 19/19 [00:15<00:00,  1.25it/s]
Epoch 1/20 | Train Loss: 0.6234 | Val Loss: 0.5891 | Val AUROC: 0.6543 | Time: 98.3s
  ✓ New best validation loss!
...
```

### 5. Load and Use Trained Models
```python
import torch
from inference_example import load_trained_model, predict_sequences, evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained C model
model, checkpoint = load_trained_model(
    '/content/drive/MyDrive/romeo/models/c_lstm.pt',
    device
)

# Load test data
test_sequences = torch.load('/content/drive/MyDrive/romeo/pretraining/c/validation/sequences.pt')
test_labels = torch.load('/content/drive/MyDrive/romeo/pretraining/c/validation/labels.pt')

# Make predictions
probabilities, predictions = predict_sequences(model, test_sequences, device)

# Evaluate
metrics = evaluate_model(model, test_sequences, test_labels, device)
print(f"AUROC: {metrics['auroc']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## Files Overview

### Training
- **`load_and_predict.py`** - Main training script for Colab
  - Loads HuggingFace embeddings automatically
  - Trains all 4 language models
  - Saves models and plots

### Inference
- **`inference_example.py`** - Examples for using trained models
  - Load saved models
  - Make predictions
  - Evaluate metrics

### Original (HPC)
- **`models/lstm.py`** - Original training script
  - For local/HPC use with manual embedding files
  - Not modified (as requested)

## Key Differences: Colab vs Original

| Feature | `load_and_predict.py` (Colab) | `models/lstm.py` (HPC) |
|---------|-------------------------------|------------------------|
| Embedding Loading | HuggingFace API | Manual `.pt` file |
| Data Paths | Google Drive | Local filesystem |
| Training | All 4 languages | Single language |
| Memory Management | Auto cleanup per language | Manual |
| Plots | Auto-generated dual plots | Single plots |
| Model Saving | Comprehensive checkpoints | Basic checkpoints |

## Configuration

Edit these variables in `load_and_predict.py` if needed:

```python
# Hyperparameters
BATCH_SIZE = 32          # Reduce if OOM errors
LEARNING_RATE = 0.001
EPOCHS = 20              # Max epochs per model
LSTM_NODES = 256         # Hidden dimension
OUTPUT_DIM = 1           # Binary classification

# Paths (adjust for your Drive structure)
c_pretraining_path = '/content/drive/MyDrive/romeo/pretraining/c'
java_pretraining_path = '/content/drive/MyDrive/romeo/pretraining/java'
csharp_pretraining_path = '/content/drive/MyDrive/romeo/pretraining/csharp'
combined_path = '/content/drive/MyDrive/romeo/pretraining/combined'
output_path = '/content/drive/MyDrive/romeo/models'
```

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 16  # or 8

# Or train one language at a time
datasets = [('c', c_pretraining_path)]  # Comment out others
```

### Missing Data
```bash
# Check file structure
!ls -R /content/drive/MyDrive/romeo/pretraining/

# Verify tensor files exist
!ls -lh /content/drive/MyDrive/romeo/pretraining/c/train/
```

### Slow Training
```python
# Verify GPU is being used
print(f"Using device: {torch.cuda.get_device_name(0)}")

# Check GPU memory
!nvidia-smi
```

### Import Errors
```bash
# Reinstall packages
!pip install --upgrade transformers torchmetrics tqdm
```

## Expected Training Times (T4 GPU)

| Language | Samples | Train Time | Total Time |
|----------|---------|------------|------------|
| C | ~1400 | ~2 min/epoch | ~30-40 min |
| Java | ~1400 | ~2 min/epoch | ~30-40 min |
| C# | ~1400 | ~2 min/epoch | ~30-40 min |
| Combined | ~4200 | ~6 min/epoch | ~90-120 min |

**Total for all 4 models: ~3-4 hours**

## Output Files

After training, you'll have:

```
/content/drive/MyDrive/romeo/models/
├── c_lstm.pt                    # C model weights + metrics
├── c_training_curves.png        # C training plots
├── java_lstm.pt                 # Java model
├── java_training_curves.png
├── csharp_lstm.pt              # C# model
├── csharp_training_curves.png
├── combined_lstm.pt            # Combined model
└── combined_training_curves.png
```

Each `.pt` file contains:
- Model state dict
- Training/validation losses
- AUROC scores
- Training time
- Full configuration
- Timestamp

## Next Steps

1. ✅ Train models using `load_and_predict.py`
2. ✅ Download models to local machine
3. ⏭️ Use `inference_example.py` to evaluate
4. ⏭️ Compare performance across languages
5. ⏭️ Fine-tune on evaluation datasets
6. ⏭️ Run experiments with different architectures
