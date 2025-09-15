# Romeo: Cross-Language Vulnerability Detection

A deep learning framework for detecting vulnerabilities in source code across multiple programming languages.

## Project Overview

Romeo is a research project focused on vulnerability detection in source code using deep learning models. The system processes functions from C and Java codebases, identifies vulnerabilities at the line level, and trains neural network models to detect these vulnerabilities in new code.

Key features:
- Cross-language vulnerability detection
- Line-level vulnerability identification
- Multiple neural network architectures (LSTM, Transformer)
- Configurable data processing pipelines
- Performance evaluation metrics

## Repository Structure

```
├── models/                 # Neural network model definitions
│   ├── lstm.py             # LSTM-based model for vulnerability detection
│   └── transformer_model.py # Self-attention transformer model
├── datasets/               # SQLite databases containing code samples
│   ├── juliet_c_10+.db     # C code vulnerability database
│   └── juliet_java_10+.db  # Java code vulnerability database
├── tensors/                # Generated tensor data for training
├── plots/                  # Visualization outputs
├── preprocessing.py        # Data processing utilities
├── preprocessing.ipynb     # Interactive preprocessing notebook
├── generate_language_matrix.py # Creates cross-language experiment sets
├── run_models.py           # Model training and evaluation script
├── environment.yml         # Conda environment specification
```

## Key Components

### Data Processing

The system processes C and Java code from the JULIET dataset and extracts functions with labeled vulnerabilities. The preprocessing pipeline:

1. Loads code functions from SQLite databases
2. Identifies vulnerable lines within each function
3. Tokenizes code using a pre-trained tokenizer
4. Creates balanced datasets with vulnerability labels
5. Splits data into train/validation/test sets
6. Generates tensor files for model training

### Models

Two main neural network architectures are implemented:

#### LSTM Classifier
- Bidirectional LSTM layers
- Pretrained embeddings
- Designed for processing code sequences line by line

#### Self-Attention Transformer
- Multi-head self-attention mechanism
- Positional encoding
- Handles complex dependencies in code

### Experiments

The project includes a comprehensive experimental framework:

- Language-specific training and testing
- Cross-language evaluations
- Combined multi-language models
- Performance metrics including loss, AUROC, and timing

## Setup and Usage

### Environment Setup

```bash
conda env create -f environment.yml
conda activate romeo
```

### Data Preparation

```bash
# Generate cross-language experiment data
python generate_language_matrix.py

# Alternatively, run preprocessing directly
python preprocessing.py
```

### Training Models

```bash
# Run models with default configurations
python run_models.py

# For hyperparameter tuning
python param_tuning.py
```

### Visualization

Several notebooks are available for visualizing results:
- `plotting.ipynb`: Result visualizations
- `model_visualization.py`: Model architecture visualization

## Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA-compatible GPU
- Transformers library
- SQLite3

## Project Status

This is an ongoing research project. Current focus areas:
- Improving cross-language generalization
- Expanding to additional programming languages
- Optimizing model performance and efficiency
- Enhancing vulnerability detection accuracy

## License
