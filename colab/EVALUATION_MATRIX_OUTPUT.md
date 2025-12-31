# Evaluation Matrix - Expected Output

This document shows the expected output format from `inference_example.py`.

## Command
```python
python inference_example.py
```

## Sample Output

```
Using device: cuda

================================================================================
EVALUATION MATRIX: Models vs Test Sets
================================================================================

================================================================================
Model: C
================================================================================
Loaded model: c
Best Val Loss: 0.3245
Final Val AUROC: 0.8567

  Testing on C test set...
    Samples: 600, Positive: 300
    Accuracy:  0.8500
    Precision: 0.8333
    Recall:    0.8667
    F1:        0.8497
    AUROC:     0.9123

  Testing on Python test set...
    Samples: 480, Positive: 240
    Accuracy:  0.6875
    Precision: 0.6842
    Recall:    0.6500
    F1:        0.6667
    AUROC:     0.7234

================================================================================
Model: JAVA
================================================================================
Loaded model: java
Best Val Loss: 0.3567
Final Val AUROC: 0.8234

  Testing on C test set...
    Samples: 600, Positive: 300
    Accuracy:  0.7000
    Precision: 0.6957
    Recall:    0.6667
    F1:        0.6809
    AUROC:     0.7456

  Testing on Python test set...
    Samples: 480, Positive: 240
    Accuracy:  0.6625
    Precision: 0.6500
    Recall:    0.6500
    F1:        0.6500
    AUROC:     0.7089

================================================================================
Model: CSHARP
================================================================================
Loaded model: csharp
Best Val Loss: 0.3423
Final Val AUROC: 0.8456

  Testing on C test set...
    Samples: 600, Positive: 300
    Accuracy:  0.7333
    Precision: 0.7273
    Recall:    0.7000
    F1:        0.7134
    AUROC:     0.7823

  Testing on Python test set...
    Samples: 480, Positive: 240
    Accuracy:  0.6792
    Precision: 0.6667
    Recall:    0.6667
    F1:        0.6667
    AUROC:     0.7145

================================================================================
Model: COMBINED
================================================================================
Loaded model: combined
Best Val Loss: 0.3101
Final Val AUROC: 0.8734

  Testing on C test set...
    Samples: 600, Positive: 300
    Accuracy:  0.8667
    Precision: 0.8571
    Recall:    0.9000
    F1:        0.8780
    AUROC:     0.9345

  Testing on Python test set...
    Samples: 480, Positive: 240
    Accuracy:  0.7458
    Precision: 0.7391
    Recall:    0.7083
    F1:        0.7234
    AUROC:     0.8012

================================================================================
EVALUATION MATRIX SUMMARY
================================================================================

--- AUROC Scores ---
Test Set      C      Python
Model                      
C         0.9123  0.7234
COMBINED  0.9345  0.8012
CSHARP    0.7823  0.7145
JAVA      0.7456  0.7089

--- Accuracy Scores ---
Test Set      C      Python
Model                      
C         0.8500  0.6875
COMBINED  0.8667  0.7458
CSHARP    0.7333  0.6792
JAVA      0.7000  0.6625

--- F1 Scores ---
Test Set      C      Python
Model                      
C         0.8497  0.6667
COMBINED  0.8780  0.7234
CSHARP    0.7134  0.6667
JAVA      0.6809  0.6500

✓ Full results saved to: /content/drive/MyDrive/romeo/models/evaluation_matrix.csv
✓ Pivot tables saved

================================================================================
Evaluation Complete!
================================================================================
```

## Generated Files

After running, these files are created in the models directory:

### 1. `evaluation_matrix.csv`
Complete results with all metrics:
```csv
Model,Test Set,Accuracy,Precision,Recall,F1,AUROC
C,C,0.8500,0.8333,0.8667,0.8497,0.9123
C,Python,0.6875,0.6842,0.6500,0.6667,0.7234
JAVA,C,0.7000,0.6957,0.6667,0.6809,0.7456
JAVA,Python,0.6625,0.6500,0.6500,0.6500,0.7089
CSHARP,C,0.7333,0.7273,0.7000,0.7134,0.7823
CSHARP,Python,0.6792,0.6667,0.6667,0.6667,0.7145
COMBINED,C,0.8667,0.8571,0.9000,0.8780,0.9345
COMBINED,Python,0.7458,0.7391,0.7083,0.7234,0.8012
```

### 2. `evaluation_matrix_auroc.csv`
AUROC scores in matrix format:
```csv
Model,C,Python
C,0.9123,0.7234
COMBINED,0.9345,0.8012
CSHARP,0.7823,0.7145
JAVA,0.7456,0.7089
```

### 3. `evaluation_matrix_accuracy.csv`
Accuracy scores in matrix format

### 4. `evaluation_matrix_f1.csv`
F1 scores in matrix format

## Key Insights from Example

1. **Best Overall**: Combined model performs best across both test sets
2. **Language-Specific**: C model excels on C test set but struggles on Python
3. **Generalization**: Combined model generalizes better to unseen languages
4. **Cross-Language**: All models show performance drop on Python (unseen during pretraining)

## Usage in Analysis

Import the CSV files into analysis tools:
```python
import pandas as pd

# Load results
df = pd.read_csv('romeo/models/evaluation_matrix.csv')

# Compare models
best_model = df.groupby('Model')['AUROC'].mean().idxmax()
print(f"Best model overall: {best_model}")

# Analyze generalization
df['Generalization'] = df.groupby('Model')['AUROC'].transform('std')
```
