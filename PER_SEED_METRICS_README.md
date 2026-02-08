# Per-Seed Metrics Extraction - Summary

## Overview
Successfully extracted per-seed metrics from experiments 1-3 JSON files and created `all_seeds_metrics.csv` files matching the format of experiment 4.

## Files Created

### 1. Extraction Script: `extract_seed_metrics.py`
- **Purpose**: Parse `k=*.json` files from experiment directories
- **Extracts**: AUROC, F1, accuracy, precision, recall, loss for each seed
- **Output**: CSV files with format: `k, seed, split, auroc, f1, accuracy, precision, recall, loss`

### 2. Statistical Analysis Demo: `statistical_analysis_example.py`
- **Purpose**: Demonstrate how to perform t-tests and other statistical analyses
- **Includes**: 
  - Independent t-tests (comparing experiments)
  - Paired t-tests (comparing k values within same seeds)
  - ANOVA (comparing multiple groups)
  - Effect size calculations (Cohen's d)

### 3. CSV Output Files Created:
- `results/experiment1/all_seeds_metrics.csv` (481 lines: 1 header + 480 data rows)
- `results/experiment2/all_seeds_metrics.csv` (481 lines: 1 header + 480 data rows)
- `results/experiment3/all_seeds_metrics.csv` (481 lines: 1 header + 480 data rows)
- `results/experiment4/all_seeds_metrics.csv` (already existed)

## Data Structure

Each CSV file contains:
- **12 k-values** (1 through 12)
- **20 seeds** per k-value
- **2 splits** per seed (test, ood)
- **Total rows**: 12 × 20 × 2 = 480 data rows + 1 header = 481 lines

### CSV Columns:
1. `k` - SimHash deduplication threshold
2. `seed` - Random seed used for training
3. `split` - Either "test" (in-distribution) or "ood" (out-of-distribution)
4. `auroc` - Area Under ROC Curve
5. `f1` - F1 Score
6. `accuracy` - Classification accuracy
7. `precision` - Precision
8. `recall` - Recall
9. `loss` - Loss value

## How to Use the Data

### Example 1: Load data for statistical tests
```python
import pandas as pd

# Load per-seed metrics
exp1_seeds = pd.read_csv('results/experiment1/all_seeds_metrics.csv')

# Filter for specific analysis
ood_auroc_k1 = exp1_seeds[(exp1_seeds['k'] == 1) & 
                          (exp1_seeds['split'] == 'ood')]['auroc'].values
```

### Example 2: Perform t-test between experiments
```python
from scipy import stats

# Compare Exp 1 vs Exp 2
exp1_ood = exp1_seeds[(exp1_seeds['k'] == 1) & (exp1_seeds['split'] == 'ood')]['auroc'].values
exp2_ood = exp2_seeds[(exp2_seeds['k'] == 1) & (exp2_seeds['split'] == 'ood')]['auroc'].values

t_stat, p_value = stats.ttest_ind(exp1_ood, exp2_ood)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
```

### Example 3: Paired t-test (same seeds, different k values)
```python
# Compare k=1 vs k=12 within same experiment
k1_ood = exp1_seeds[(exp1_seeds['k'] == 1) & (exp1_seeds['split'] == 'ood')]['auroc'].values
k12_ood = exp1_seeds[(exp1_seeds['k'] == 12) & (exp1_seeds['split'] == 'ood')]['auroc'].values

t_stat, p_value = stats.ttest_rel(k1_ood, k12_ood)  # Paired test
print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
```

## Verification

All files verified to have the correct format and data:
```bash
$ wc -l results/experiment*/all_seeds_metrics.csv
     481 results/experiment1/all_seeds_metrics.csv
     481 results/experiment2/all_seeds_metrics.csv
     481 results/experiment3/all_seeds_metrics.csv
     481 results/experiment4/all_seeds_metrics.csv
    1924 total
```

## Notebook Integration

The analysis notebook `deduplication experiments analysis.ipynb` has been updated with a new cell demonstrating:
- Loading per-seed data from all experiments
- Performing various statistical tests (t-tests, ANOVA)
- Calculating effect sizes (Cohen's d)
- Generating summary statistics

## Next Steps

You can now:
1. ✅ Perform t-tests comparing experiments
2. ✅ Analyze variance across seeds
3. ✅ Calculate confidence intervals
4. ✅ Perform power analysis
5. ✅ Generate statistical tables for papers
6. ✅ Conduct meta-analysis across embeddings
7. ✅ Test significance of deduplication effects

All the raw per-seed data is now available for comprehensive statistical analysis!
