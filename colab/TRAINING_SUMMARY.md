# LSTM Model Training Summary

## Overview
Updated the LSTM vulnerability detection model from multi-label line prediction to binary sequence classification (vulnerable vs. secure).

## Key Changes

### 1. Model Architecture (`models/lstm.py`)
**BEFORE:** Predicted vulnerability for each line (150 outputs per sequence)
**AFTER:** Binary classification for entire sequence (1 output: vulnerable or secure)

**Changes:**
- `OUTPUT_DIM`: Changed from `NUM_SENTENCES = 150` to `OUTPUT_DIM = 1`
- `MAX_SEQ_LENGTH`: Set to 4096 tokens
- `BATCH_SIZE`: Increased from 20 to 32
- Model output: Now returns single probability per sequence
- Labels: Binary (0 = secure, 1 = vulnerable)

### 2. Training Function
**BEFORE:**
```python
predictions = predictions.view(-1, NUM_SENTENCES).float()
batch_labels = batch_labels.view(-1, NUM_SENTENCES).float()
```

**AFTER:**
```python
predictions = predictions.squeeze(1)  # [batch_size, 1] -> [batch_size]
batch_labels = batch_labels.float()   # Binary labels
```

### 3. Evaluation Function
- Removed extra sigmoid (already applied in forward pass)
- Updated AUROC computation for binary classification
- Simplified label handling

### 4. New Training Script (`load_and_predict.py`)

**Purpose:** Google Colab script to train models on all language datasets

**Features:**
- Loads embeddings from HuggingFace `aiXcoder/aixcoder-7b-base`
- Trains separate models for C, Java, C#, and combined datasets
- Implements early stopping (patience = 3 epochs)
- Saves models with full metrics and configuration
- Generates training curve visualizations
- Memory-efficient (clears GPU after each language)

**Key Functions:**
- `load_tensors()`: Loads sequences and labels from pretraining directories
- `create_dataloaders()`: Creates PyTorch DataLoaders
- `train_epoch()`: Single epoch training
- `evaluate()`: Validation with AUROC computation
- `train_model()`: Complete training pipeline for one language
- `main()`: Trains all language models sequentially

### 5. Data Loading

**Directory Structure:**
```
tensors/TIMESTAMP/pretraining/
├── c/
│   ├── train/
│   │   ├── sequences.pt
│   │   └── labels.pt
│   └── validation/
│       ├── sequences.pt
│       └── labels.pt
├── java/
├── csharp/
└── combined/
```

**For Google Colab:**
```
/content/drive/MyDrive/romeo/pretraining/
└── [same structure as above]
```

## Usage

### Local Training (Original LSTM)
```bash
cd /Users/aeg00011/Romeo/models
python lstm.py
```
**Requirements:**
- Update paths to point to tensor directories
- Requires `aix3-7b-base (1).pt` file for embeddings

### Google Colab Training (New Script)
```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Install packages
!pip install transformers torchmetrics

# 3. Run training
!python load_and_predict.py
```

**Automatic Features:**
- Downloads embeddings from HuggingFace (no manual file needed)
- Trains all 4 models (C, Java, C#, Combined)
- Saves to `/content/drive/MyDrive/romeo/models/`
- Generates plots automatically

## Model Output

Each saved model (`.pt` file) contains:
```python
{
    'model_state_dict': state_dict,
    'best_val_loss': float,
    'final_val_auroc': float,
    'train_losses': list,
    'valid_losses': list,
    'valid_aurocs': list,
    'training_time': float (seconds),
    'timestamp': ISO format string,
    'config': {
        'vocab_size': 49152,
        'embedding_dim': 4096,
        'hidden_dim': 256,
        'output_dim': 1,
        'n_layers': 2,
        'bidirectional': True,
        'dropout': 0.5,
        'language': str
    }
}
```

## Loading a Saved Model

```python
import torch
from models.lstm import LSTMClassifier

# Load checkpoint
checkpoint = torch.load('path/to/model.pt')

# Initialize model
model = LSTMClassifier(
    vocab_size=checkpoint['config']['vocab_size'],
    embedding_dim=checkpoint['config']['embedding_dim'],
    hidden_dim=checkpoint['config']['hidden_dim'],
    output_dim=checkpoint['config']['output_dim'],
    n_layers=checkpoint['config']['n_layers'],
    batch_first=True,
    bidirectional=checkpoint['config']['bidirectional'],
    dropout=checkpoint['config']['dropout'],
    pretrained_weights=word_vectors  # Load separately
)

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(sequences).squeeze(1)
    binary_preds = (predictions > 0.5).long()
```

## Performance Metrics

The script tracks:
- **Training Loss**: BCELoss per epoch
- **Validation Loss**: BCELoss on validation set
- **Validation AUROC**: Area under ROC curve (vulnerability detection)
- **Training Time**: Total and per-epoch timing
- **Early Stopping**: Stops if no improvement for 3 epochs

## Files Modified

1. ✅ `models/lstm.py` - Updated for binary classification
2. ✅ `load_and_predict.py` - New Colab training script
3. ✅ `preprocessing.py` - Added `load_data_from_db_eval()` for evaluation datasets
4. ✅ `pretraining_generation.py` - Complete implementation for tensor generation

## Next Steps

1. Upload tensors to Google Drive
2. Run `load_and_predict.py` in Colab
3. Monitor training progress (loss and AUROC)
4. Download trained models
5. Evaluate on test sets (evaluation tensors)
6. Compare performance across languages
