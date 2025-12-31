# LSTM Vulnerability Detection Model - Technical Documentation

## Overview

This document describes the bidirectional LSTM classifier with attention mechanism used for binary vulnerability detection in source code. The model predicts whether a code sequence is vulnerable (1) or secure (0).

---

## Model Architecture

### High-Level Design

```
Input Sequence (Token IDs)
    ↓
Embedding Layer (Pretrained)
    ↓
Dropout (p=0.5)
    ↓
Bidirectional LSTM (2 layers)
    ↓
Attention Mechanism
    ↓
Dropout (p=0.5)
    ↓
Fully Connected Layer
    ↓
Sigmoid Activation
    ↓
Binary Prediction [0, 1]
```

### Layer-by-Layer Architecture

#### 1. **Embedding Layer**
- **Type**: `nn.Embedding`
- **Vocabulary Size**: 49,152 tokens
- **Embedding Dimension**: 4,096
- **Pretrained**: Yes (aiXcoder-7b-base token embeddings)
- **Trainable**: Yes (fine-tuned during training)
- **Purpose**: Converts token IDs into dense vector representations

#### 2. **LSTM Layer**
- **Type**: `nn.LSTM` (Bidirectional)
- **Hidden Dimension**: 256 units per direction (512 total)
- **Number of Layers**: 2
- **Bidirectional**: True
- **Dropout**: 0.5 (between LSTM layers)
- **Batch First**: True
- **Output Shape**: [batch_size, sequence_length, 512]
- **Purpose**: Captures sequential dependencies in both forward and backward directions

#### 3. **Attention Mechanism**
- **Type**: Additive attention
- **Implementation**: `nn.Linear(hidden_dim * 2, 1)` with softmax
- **Input**: LSTM outputs [batch_size, seq_length, 512]
- **Output**: Attention weights [batch_size, seq_length, 1]
- **Attended Output**: Weighted sum of LSTM outputs [batch_size, 512]
- **Purpose**: Focuses on the most relevant parts of the sequence for vulnerability detection

#### 4. **Fully Connected Layer**
- **Type**: `nn.Linear`
- **Input Dimension**: 512 (bidirectional LSTM output)
- **Output Dimension**: 1 (binary classification)
- **Activation**: Sigmoid (applied in forward pass)
- **Purpose**: Maps attended features to binary vulnerability probability

---

## Hyperparameters

### Model Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Vocabulary Size** | 49,152 | Size of token vocabulary from aiXcoder tokenizer |
| **Embedding Dimension** | 4,096 | Dimension of pretrained token embeddings |
| **LSTM Hidden Units** | 256 | Hidden units per direction (512 bidirectional) |
| **LSTM Layers** | 2 | Number of stacked LSTM layers |
| **Output Dimension** | 1 | Binary classification output |
| **Bidirectional** | True | LSTM processes sequences in both directions |
| **Dropout Rate** | 0.5 | Applied after embedding and before FC layer |
| **Batch First** | True | Input format: [batch, sequence, feature] |

### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Optimizer** | Adam | Adaptive learning rate optimizer |
| **Learning Rate** | 0.001 | Initial learning rate |
| **Loss Function** | BCELoss | Binary Cross Entropy Loss |
| **Batch Size** | 32 | Number of samples per batch |
| **Max Epochs** | 20 (HPC) / 30 (Colab) | Maximum training epochs |
| **Gradient Clipping** | 1.0 | Max gradient norm to prevent exploding gradients |
| **Early Stopping Patience** | 5 (HPC) / 8 (Colab) | Epochs without improvement before stopping |
| **Max Sequence Length** | 4,096 | Maximum tokens per code sequence |
| **Min Sequence Length** | 32 | Minimum tokens required |

### Data Split Ratios

#### Pretraining Datasets (Juliet)
- **Training**: 70%
- **Validation**: 30%

#### Evaluation Datasets (Devign, BugsInPy)
- **Split Mode**:
  - Training: 60%
  - Validation: 20%
  - Test: 20%
- **Full Mode**: 100% (no splits, entire dataset available)

---

## Training Process

### 1. **Data Preparation**
- Code sequences tokenized using aiXcoder tokenizer
- Sequences padded/truncated to max 4,096 tokens
- Binary labels: 1 (vulnerable), 0 (secure)
- Classes balanced during sampling

### 2. **Forward Pass**
```python
1. Input: Token IDs [batch_size, seq_length]
2. Embedding: [batch_size, seq_length, 4096]
3. Dropout: Applied to embeddings
4. LSTM: [batch_size, seq_length, 512] (bidirectional)
5. Attention: Compute weights and weighted sum → [batch_size, 512]
6. Dropout: Applied to attended output
7. FC Layer: [batch_size, 512] → [batch_size, 1]
8. Sigmoid: [batch_size, 1] with values in [0, 1]
```

### 3. **Loss Calculation**
- **Binary Cross Entropy Loss**:
  ```
  BCE(y, ŷ) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
  ```
- Predictions are probabilities from sigmoid
- Labels are binary (0 or 1)

### 4. **Optimization Step**
1. Zero gradients: `optimizer.zero_grad()`
2. Compute loss: `loss = criterion(predictions, labels)`
3. Backward pass: `loss.backward()`
4. **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
5. Update weights: `optimizer.step()`

### 5. **Early Stopping**
- Monitors validation loss
- Stops training if no improvement for `PATIENCE` consecutive epochs
- Saves best model checkpoint based on lowest validation loss

---

## Evaluation Metrics

### Training Metrics
- **Training Loss**: BCE loss on training set
- **Validation Loss**: BCE loss on validation set
- **Validation AUROC**: Area Under ROC Curve on validation set
- **Epoch Times**: Training and evaluation duration per epoch

### Inference Metrics (Multi-Seed Evaluation)
- **Accuracy**: Correct predictions / Total predictions
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1 Score**: Harmonic mean of precision and recall

Each model evaluated 5 times with different random seeds, reporting:
- **Mean**: Average across runs
- **Standard Deviation**: Variability across runs
- **Min/Max**: Range of performance
- **Range**: Max - Min

---

## Model Checkpointing

### During Training
Each epoch saves a checkpoint with:
- Model state dict
- Optimizer state
- Current loss
- Epoch number
- Training duration

### Final Model
Saved with comprehensive metadata:
```python
{
    'model_state_dict': <trained weights>,
    'test_loss': <final test loss>,
    'test_auroc': <test AUROC score>,
    'training_time': <total training seconds>,
    'timestamp': <ISO format timestamp>,
    'config': {
        'vocab_size': 49152,
        'embedding_dim': 4096,
        'hidden_dim': 256,
        'output_dim': 1,
        'n_layers': 2,
        'bidirectional': True,
        'dropout': 0.5,
        'patience': 5 or 8,
        'gradient_clip': 1.0,
        'learning_rate': 0.001
    }
}
```

---

## Training Environments

### HPC Environment (`models/lstm.py`)
- **Max Epochs**: 20
- **Patience**: 5
- **Input**: Tensor files loaded directly from disk
- **Output**: Checkpoints saved per epoch + final model
- **Plots**: Loss curves, AUROC curves, timing plots

### Google Colab Environment (`load_and_predict.py`)
- **Max Epochs**: 30
- **Patience**: 8
- **Input**: Embeddings loaded from HuggingFace
- **Training Sets**: C, Java, C#, and combined
- **Output**: Models saved to Google Drive
- **Memory Management**: Cleans up HF model after loading embeddings

---

## Pretrained Embeddings

### Source
- **Model**: aiXcoder-7b-base
- **Provider**: HuggingFace Transformers
- **Type**: Token embeddings from large language model
- **Dimensionality**: 4,096
- **Vocabulary**: 49,152 tokens

### Loading Methods
1. **HPC**: Load from `.pt` file (`aix3-7b-base (1).pt`)
2. **Colab**: Download via `AutoModelForCausalLM.from_pretrained()`

### Fine-Tuning
- Embeddings are **trainable** during training
- Initialized with pretrained weights
- Adapted to vulnerability detection task through backpropagation

---

## Key Design Decisions

### 1. **Bidirectional LSTM**
- **Rationale**: Vulnerabilities may depend on context before and after a code location
- **Benefit**: Captures dependencies in both directions of the sequence

### 2. **Attention Mechanism**
- **Rationale**: Not all parts of code equally important for vulnerability detection
- **Benefit**: Model learns to focus on suspicious code patterns
- **Implementation**: Weighted sum of LSTM outputs based on learned attention weights

### 3. **Gradient Clipping (1.0)**
- **Rationale**: Prevents exploding gradients common in RNN training
- **Benefit**: Stabilizes training on long sequences (up to 4,096 tokens)

### 4. **Dropout (0.5)**
- **Rationale**: Large embedding dimension (4,096) risks overfitting
- **Benefit**: Regularization improves generalization
- **Application**: After embedding and before final FC layer

### 5. **Early Stopping**
- **Rationale**: Prevents overfitting to training data
- **Benefit**: Model trained until validation loss plateaus
- **Implementation**: Monitors validation loss with patience of 5-8 epochs

### 6. **Multi-Seed Evaluation**
- **Rationale**: Account for random initialization variance
- **Benefit**: Robust performance estimates with confidence intervals
- **Implementation**: 5 runs with different seeds, report mean ± std

---

## Performance Considerations

### Memory Requirements
- **Embedding Layer**: 49,152 × 4,096 × 4 bytes ≈ 805 MB
- **Batch Processing**: 32 sequences × 4,096 tokens × 4 bytes ≈ 525 KB per batch
- **LSTM States**: Hidden states maintained for bidirectional processing

### Computational Complexity
- **Time Complexity**: O(sequence_length × hidden_dim²) for LSTM
- **Attention Overhead**: O(sequence_length × hidden_dim) additional
- **Typical Training Time**: Varies by dataset size and hardware
  - HPC: Tracked per epoch with timedelta output
  - Colab: Similar tracking with Drive I/O overhead

### Optimization Strategies
- **Batch Processing**: Groups of 32 sequences processed in parallel
- **GPU Acceleration**: All tensors moved to CUDA device if available
- **Gradient Accumulation**: Single backward pass per batch (no accumulation)
- **Memory Cleanup**: Explicit `torch.cuda.empty_cache()` in Colab script

---

## Usage Example

### Loading a Trained Model
```python
import torch
from models.lstm import LSTMClassifier

# Load checkpoint
checkpoint = torch.load('C_only.pt')
config = checkpoint['config']

# Initialize model
model = LSTMClassifier(
    vocab_size=config['vocab_size'],
    embedding_dim=config['embedding_dim'],
    hidden_dim=config['hidden_dim'],
    output_dim=config['output_dim'],
    n_layers=config['n_layers'],
    batch_first=True,
    bidirectional=config['bidirectional'],
    dropout=config['dropout'],
    pretrained_weights=word_vectors  # Load separately
)

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Making Predictions
```python
# Prepare input (tokenized sequence)
tokens = tokenizer.encode(code_snippet, max_length=4096, truncation=True)
input_tensor = torch.tensor([tokens]).long()

# Predict
with torch.no_grad():
    prediction = model(input_tensor)
    is_vulnerable = prediction.item() > 0.5
```

---

## References

- **Tokenizer**: aiXcoder/aixcoder-7b-base (HuggingFace)
- **Framework**: PyTorch
- **Metrics**: torchmetrics library
- **Datasets**: Juliet Test Suite, Devign, BugsInPy

---

## Version Information

- **Model Version**: Binary classification (v2)
- **Architecture**: Bidirectional LSTM with attention
- **Last Updated**: December 2024
- **Python Version**: 3.11.2
- **PyTorch Version**: Compatible with CUDA

---

## Contact & Maintenance

For questions about model architecture, hyperparameters, or training procedures, refer to:
- `models/lstm.py` - HPC training script
- `load_and_predict.py` - Colab training script
- `inference_example.py` - Evaluation and inference
- `pretraining_generation.py` - Data pipeline configuration

