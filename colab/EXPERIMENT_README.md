# Multilingual Vulnerability Detection Experiment

This experiment evaluates the performance of LSTM models trained on different programming languages (C, Java, and multilingual) for vulnerability detection.

## Structure

- `train_and_evaluate.py`: Main script for training and evaluating models on different language configurations
- `models/`: Directory where trained models are saved
- `results/`: Directory where evaluation results are saved
- `plots/`: Directory where training metrics plots are saved
- `checkpoints/`: Directory for model checkpoints during training

## Tensor Structure

The experiment uses tensors organized in the following directory structure:

```
tensors/Experiment_1/
  c_only_seed42/           # C language only training
    20250911_182225_seed42/
      c/                  # C language data
        train_sequences.pt
        train_labels.pt
        val_sequences.pt
        val_labels.pt
        test_sequences.pt
        test_labels.pt
      java/               # Java language data
        ...
      splits/             # Multilingual data
        ...
  java_only_seed43/        # Java language only training
    ...
  combined_seed46/         # Multilingual training
    ...
```

## Usage

### Running the Full Experiment

To train all models and evaluate them on all test sets, simply run:

```bash
python train_and_evaluate.py
```

This will:
1. Train three models (C only, Java only, multilingual)
2. Evaluate each model on C, Java, and multilingual test sets
3. Generate a performance matrix showing the AUROC scores for each model-test pair
4. Save results and plots

### Training a Specific Model

To train only a specific model:

```bash
python train_and_evaluate.py --train c_only
```

Options for the `--train` argument:
- `c_only`: Train model on C language only
- `java_only`: Train model on Java language only
- `multilingual`: Train model on both C and Java languages
- `all`: Train all three models

### Evaluating Models

To evaluate all trained models without retraining:

```bash
python train_and_evaluate.py --evaluate
```

### Using Custom Pretrained Weights

To specify a custom path for pretrained weights:

```bash
python train_and_evaluate.py --weights path/to/weights.pt
```

## Results

After running the experiment, you'll find:

1. Trained models in the `models/` directory:
   - `c_only_model_best.pt`
   - `java_only_model_best.pt`
   - `multilingual_model_best.pt`

2. Performance matrix in the `results/` directory:
   - Shows AUROC scores for each model on each test set
   - Format: `performance_matrix_TIMESTAMP.csv`

3. Training metric plots in the `plots/` directory:
   - Loss curves
   - AUROC scores
   - Training times

## Performance Matrix

The performance matrix shows the AUROC scores for each model when tested on different language datasets:

| Training Dataset | Performance on C | Performance on Java | Performance on Multilingual |
|------------------|------------------|--------------------|-----------------------------|
| C only           |                  |                    |                             |
| Java only        |                  |                    |                             |
| Multilingual     |                  |                    |                             |

This matrix helps to evaluate how well models trained on one language generalize to others, and whether multilingual training offers advantages.