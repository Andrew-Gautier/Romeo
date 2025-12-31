# Dynamic Thresholding Quick Reference

## TL;DR - What Changed

**Problem**: Model too cautious → many false negatives (missed vulnerabilities)

**Solution**: Dynamic thresholding → automatically lowers threshold to catch more vulnerabilities

**Usage**: Add 2 parameters to your evaluation call:
```python
results = create_evaluation_matrix(
    models_dir, eval_base_dir, device,
    use_auto_threshold=True,  # NEW: Enable dynamic thresholding
    min_recall=0.90           # NEW: Require 90% vulnerability detection
)
```

---

## Key Functions Added

### 1. Find Optimal Threshold
```python
threshold, metrics, analysis = find_optimal_threshold(
    model, sequences, labels,
    metric='f2',        # Optimize F2 (favors recall)
    min_recall=0.95     # Minimum recall requirement
)
```

### 2. Evaluate with Custom Threshold
```python
metrics = evaluate_model(
    model, sequences, labels,
    threshold=0.35,  # Use custom threshold instead of 0.5
    return_probabilities=True
)
```

### 3. Auto-Threshold Evaluation Matrix
```python
results = create_evaluation_matrix(
    models_dir, eval_base_dir, device,
    use_auto_threshold=True,  # Automatically find best threshold
    min_recall=0.90           # Ensure 90%+ recall
)
```

---

## Threshold Recommendations

| min_recall | Use Case | Expected Results |
|-----------|----------|------------------|
| **0.95** | Critical systems, compliance | Catch 95%+ vulnerabilities, high false positives |
| **0.90** | General security apps (RECOMMENDED) | Catch 90%+ vulnerabilities, moderate false positives |
| **0.85** | Lower risk, limited review resources | Catch 85%+ vulnerabilities, lower false positives |

---

## New Metrics Explained

| Metric | What It Means | Goal |
|--------|---------------|------|
| **Recall** | % of vulnerabilities detected | **Maximize** (most important for security) |
| **F2 Score** | Weighted metric favoring recall | **Maximize** (better than F1 for security) |
| **False Negatives** | # of missed vulnerabilities | **Minimize** (critical security metric) |
| **False Positives** | # of safe code flagged | Accept higher for security |
| **Precision** | % of flags that are real vulnerabilities | Balance (don't optimize solely) |

---

## Example: Before vs After

### Before (0.5 threshold)
```
Recall: 0.52        ⚠️ Missing 48% of vulnerabilities
Precision: 0.88     ✅ Very precise but too cautious
F1: 0.65
False Negatives: 48 ⚠️ TOO HIGH FOR SECURITY
```

### After (auto-threshold = 0.28)
```
Recall: 0.93        ✅ Catching 93% of vulnerabilities
Precision: 0.48     ⚡ Lower but acceptable for security
F2: 0.79            ✅ Better security-focused score
False Negatives: 7  ✅ Much safer!
```

**Result**: 85% reduction in missed vulnerabilities (48 → 7)

---

## Quick Decision Matrix

**Should I use auto-thresholding?**

| Scenario | Use Auto-Threshold? | min_recall |
|----------|-------------------|-----------|
| Security-critical application | ✅ YES | 0.90-0.95 |
| Production vulnerability scanner | ✅ YES | 0.90 |
| Research/experimentation | ✅ YES | 0.85-0.90 |
| Limited review resources | ⚡ MAYBE | 0.85 |
| General ML task (non-security) | ❌ NO | N/A |

---

## Output Files Generated

| File | Description |
|------|-------------|
| `evaluation_matrix_summary.csv` | Mean & std for all metrics |
| `evaluation_thresholds.csv` | **Thresholds used per model** |
| `evaluation_matrix_recall_mean.csv` | Recall scores (security focus) |
| `evaluation_matrix_f2_mean.csv` | F2 scores (security focus) |
| `evaluation_matrix_false_negatives.csv` | **Missed vulnerabilities count** |
| `evaluation_matrix_detailed.csv` | Per-run detailed results |

---

## Common Patterns

### Pattern 1: Quick Single Model Test
```python
# Load and evaluate with optimal threshold
model, _ = load_trained_model('model.pt', device)
test_seq = torch.load('test/sequences.pt')
test_labels = torch.load('test/labels.pt')

threshold, _, _ = find_optimal_threshold(model, test_seq, test_labels, min_recall=0.90)
metrics = evaluate_model(model, test_seq, test_labels, threshold=threshold)

print(f"Threshold: {threshold:.3f}, Recall: {metrics['recall']:.3f}, FN: {metrics['false_negatives']}")
```

### Pattern 2: Compare Threshold Strategies
```python
thresholds = [0.3, 0.4, 0.5]
for t in thresholds:
    m = evaluate_model(model, sequences, labels, threshold=t)
    print(f"@{t:.1f}: Recall={m['recall']:.3f}, FN={m['false_negatives']}")
```

### Pattern 3: Full Evaluation with Auto-Threshold
```python
results = create_evaluation_matrix(
    models_dir, eval_base_dir, device,
    num_runs=5,
    use_auto_threshold=True,
    min_recall=0.90
)
# Check: evaluation_thresholds.csv for optimal thresholds per model
```

---

## Troubleshooting

**"Warning: No threshold achieves minimum recall"**
- Your min_recall is too high for this model
- Lower min_recall (try 0.85 or 0.80)
- Or: Model needs retraining with better data

**Precision is very low (<0.3)**
- Expected with very high recall requirements
- Review false positive patterns
- Consider slightly lower min_recall

**Too many false positives to review**
- Increase min_recall (e.g., 0.90 → 0.85)
- Filter by probability score (only review p > 0.6)
- Use ensemble methods or model improvements

---

## Remember

🔒 **Security First**: In vulnerability detection, missing a bug is worse than a false alarm

📊 **Watch Recall & F2**: These are your primary metrics, not accuracy or F1

⚖️ **Accept Trade-offs**: Lower precision is the cost of higher recall

🎯 **Start with 0.90**: Good balance for most security applications

📈 **Monitor False Negatives**: This is the most critical metric for security

---

For detailed explanation, see `DYNAMIC_THRESHOLDING_GUIDE.md`
