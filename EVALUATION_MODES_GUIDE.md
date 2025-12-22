# Pretraining Generation Modes Guide

## Overview

The `pretraining_generation.py` script now supports two different modes for creating evaluation tensors:

1. **Split Mode** (default) - Divides evaluation data into train/validation/test splits
2. **Full Mode** - Saves all evaluation data as a single dataset without splits

## Command-Line Usage

### Basic Usage (Split Mode - Default)

```bash
python pretraining_generation.py
```

or explicitly:

```bash
python pretraining_generation.py --eval-mode split
```

### Full Dataset Mode (No Splits)

```bash
python pretraining_generation.py --eval-mode full
```

### Additional Options

```bash
python pretraining_generation.py \
    --eval-mode full \
    --pretraining-samples 5000 \
    --eval-samples 5000
```

## Mode Comparison

### Split Mode (`--eval-mode split`)

**What it does:**
- Splits evaluation data into train (60%), validation (20%), and test (20%)
- Uses stratified splitting to maintain class balance
- Good for traditional ML workflows with separate training and evaluation

**Output structure:**
```
tensors/TIMESTAMP/evaluation/
├── c/
│   ├── train/
│   │   ├── sequences.pt
│   │   └── labels.pt
│   ├── validation/
│   │   ├── sequences.pt
│   │   └── labels.pt
│   └── test/
│       ├── sequences.pt
│       └── labels.pt
├── python/
│   ├── train/
│   ├── validation/
│   └── test/
└── combined/
    ├── train/
    │   ├── sequences.pt
    │   ├── labels.pt
    │   └── languages.pt
    ├── validation/
    └── test/
```

**Use cases:**
- Fine-tuning pretrained models on evaluation data
- Traditional train/val/test workflow
- Hyperparameter tuning with validation set
- Final model evaluation on held-out test set

### Full Mode (`--eval-mode full`)

**What it does:**
- Saves all evaluation data as a single unified dataset
- No splitting - all samples in one folder
- Good for inference-only scenarios or external splitting

**Output structure:**
```
tensors/TIMESTAMP/evaluation/
├── c/
│   └── full/
│       ├── sequences.pt
│       └── labels.pt
├── python/
│   └── full/
│       ├── sequences.pt
│       └── labels.pt
└── combined/
    └── full/
        ├── sequences.pt
        ├── labels.pt
        └── languages.pt
```

**Use cases:**
- Pure inference/evaluation on pretrained models
- Custom splitting logic outside the script
- Cross-validation schemes
- Maximum data availability for evaluation
- Benchmarking pretrained models

## Output Comparison

### Split Mode Output

```
PHASE 2: Creating Evaluation Tensors
Mode: Split (train/val/test)
================================================================================

Loading c data from datasets/devign.db...
  Loaded train - Sequences: torch.Size([6000, 4096]), Labels: torch.Size([6000])
  Loaded validation - Sequences: torch.Size([2000, 4096]), Labels: torch.Size([2000])
  Loaded test - Sequences: torch.Size([2000, 4096]), Labels: torch.Size([2000])

Train: 6000, Validation: 2000, Test: 2000 samples

Tokenizing data...
Saved c training data: 6000 samples
Saved c validation data: 2000 samples
Saved c test data: 2000 samples
```

### Full Mode Output

```
PHASE 2: Creating Evaluation Tensors
Mode: Full dataset (no splits)
================================================================================

Loading c data from datasets/devign.db...
Using all 10000 samples as single dataset (no splits)

Tokenizing data...
Saved c full dataset: 10000 samples
```

## Complete Examples

### Example 1: Default Training Pipeline

Generate both pretraining and evaluation tensors with standard splits:

```bash
python pretraining_generation.py
```

This creates:
- Pretraining: C, Java, C# with train/validation splits
- Evaluation: C, Python with train/validation/test splits

### Example 2: Full Evaluation for Benchmarking

Generate tensors for benchmarking pretrained models (no training on eval data):

```bash
python pretraining_generation.py --eval-mode full
```

Use the resulting tensors with `inference_example.py` to evaluate pretrained models.

### Example 3: Smaller Dataset for Quick Testing

```bash
python pretraining_generation.py \
    --eval-mode split \
    --pretraining-samples 1000 \
    --eval-samples 1000
```

### Example 4: Large Scale Evaluation Dataset

```bash
python pretraining_generation.py \
    --eval-mode full \
    --pretraining-samples 10000 \
    --eval-samples 50000
```

## Metadata Tracking

The pipeline automatically saves metadata including the mode used:

```json
{
  "timestamp": "20241208_143022",
  "eval_mode": "split",
  "pretraining_sample_size": 10000,
  "eval_sample_size": 10000,
  "evaluation_stats": {
    "split_mode": true,
    "train_size": 6000,
    "val_size": 2000,
    "test_size": 2000
  }
}
```

or for full mode:

```json
{
  "timestamp": "20241208_143022",
  "eval_mode": "full",
  "pretraining_sample_size": 10000,
  "eval_sample_size": 10000,
  "evaluation_stats": {
    "split_mode": false,
    "total_size": 10000
  }
}
```

## Loading Tensors Based on Mode

### Loading Split Mode Tensors

```python
import torch

# Load test set for evaluation
test_sequences = torch.load('tensors/TIMESTAMP/evaluation/c/test/sequences.pt')
test_labels = torch.load('tensors/TIMESTAMP/evaluation/c/test/labels.pt')

# Or load training set for fine-tuning
train_sequences = torch.load('tensors/TIMESTAMP/evaluation/c/train/sequences.pt')
train_labels = torch.load('tensors/TIMESTAMP/evaluation/c/train/labels.pt')
```

### Loading Full Mode Tensors

```python
import torch

# Load full dataset
full_sequences = torch.load('tensors/TIMESTAMP/evaluation/c/full/sequences.pt')
full_labels = torch.load('tensors/TIMESTAMP/evaluation/c/full/labels.pt')

# Custom split if needed
from sklearn.model_selection import train_test_split
train_seq, test_seq, train_lab, test_lab = train_test_split(
    full_sequences, full_labels, test_size=0.2, random_state=42
)
```

## Integration with inference_example.py

### For Split Mode

The evaluation matrix script works directly:

```python
python inference_example.py
```

It will automatically find and use the test sets.

### For Full Mode

Modify paths in `inference_example.py`:

```python
# Change from:
test_sequences = torch.load('.../evaluation/c/test/sequences.pt')

# To:
test_sequences = torch.load('.../evaluation/c/full/sequences.pt')
```

## Command-Line Help

```bash
python pretraining_generation.py --help
```

Output:
```
usage: pretraining_generation.py [-h] [--eval-mode {split,full}]
                                  [--pretraining-samples PRETRAINING_SAMPLES]
                                  [--eval-samples EVAL_SAMPLES]

Generate pretraining and evaluation tensors for vulnerability detection

optional arguments:
  -h, --help            show this help message and exit
  --eval-mode {split,full}
                        Evaluation tensor mode: "split" for train/val/test
                        splits (60/20/20), "full" for single dataset with no
                        splits (default: split)
  --pretraining-samples PRETRAINING_SAMPLES
                        Number of samples per class for pretraining datasets
                        (default: 10000)
  --eval-samples EVAL_SAMPLES
                        Number of samples per class for evaluation datasets
                        (default: 10000)

Examples:
  # Generate with split evaluation (train/val/test) - default
  python pretraining_generation.py
  python pretraining_generation.py --eval-mode split
  
  # Generate with full evaluation (no splits)
  python pretraining_generation.py --eval-mode full
```

## Best Practices

### When to Use Split Mode

✅ Fine-tuning models on evaluation data
✅ Need validation set for hyperparameter tuning
✅ Traditional ML workflow with separate test set
✅ Want to prevent data leakage during training

### When to Use Full Mode

✅ Evaluating pretrained models (no additional training)
✅ Benchmarking multiple models on same data
✅ Need maximum data for evaluation
✅ Custom splitting logic required
✅ Cross-validation experiments

## Directory Structure Summary

```
tensors/
└── 20241208_143022/
    ├── pipeline_metadata.json
    ├── pretraining/
    │   ├── c/
    │   │   ├── train/
    │   │   └── validation/
    │   ├── java/
    │   ├── csharp/
    │   └── combined/
    └── evaluation/
        ├── c/
        │   ├── [train/validation/test/]  (split mode)
        │   └── [full/]                   (full mode)
        ├── python/
        └── combined/
```

## Migration Guide

If you have existing scripts expecting split mode:

```python
# Old code (split mode)
test_path = f'{base_dir}/evaluation/c/test'

# Make it mode-agnostic
import os
test_path = f'{base_dir}/evaluation/c/test' if os.path.exists(f'{base_dir}/evaluation/c/test') \
            else f'{base_dir}/evaluation/c/full'
```

Or check metadata:

```python
import json
with open(f'{base_dir}/pipeline_metadata.json') as f:
    metadata = json.load(f)
    
if metadata['eval_mode'] == 'split':
    test_path = f'{base_dir}/evaluation/c/test'
else:
    test_path = f'{base_dir}/evaluation/c/full'
```
