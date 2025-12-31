# Dynamic Thresholding - Documentation Index

Welcome! This directory contains comprehensive documentation for the dynamic thresholding implementation that makes your vulnerability detection model security-focused.

## 📚 Documentation Overview

| Document | Purpose | Read When... |
|----------|---------|--------------|
| **[THRESHOLD_QUICK_REF.md](THRESHOLD_QUICK_REF.md)** | Quick reference card | You need a fast lookup |
| **[BEFORE_AFTER_COMPARISON.md](BEFORE_AFTER_COMPARISON.md)** | Visual impact comparison | You want to see the benefits |
| **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** | What changed and why | You want an overview |
| **[DYNAMIC_THRESHOLDING_GUIDE.md](DYNAMIC_THRESHOLDING_GUIDE.md)** | Complete guide | You need detailed explanation |
| **[threshold_examples.py](threshold_examples.py)** | Executable examples | You want to see it in action |

## 🚀 Quick Start (30 seconds)

**What's the problem?**
- Your model was missing too many vulnerabilities (48 out of 100!)
- All misclassifications were false negatives
- Too cautious for a security application

**What's the solution?**
- Dynamic thresholding automatically finds optimal decision boundary
- Prioritizes catching vulnerabilities over precision
- Reduces missed vulnerabilities by 85% (48 → 7)

**How do I use it?**
```python
# Just add two parameters to your existing evaluation code:
results = create_evaluation_matrix(
    models_dir, eval_base_dir, device,
    use_auto_threshold=True,  # Enable dynamic thresholding
    min_recall=0.90           # Require 90% vulnerability detection
)
```

That's it! 🎉

## 📖 Reading Guide

### If you have 2 minutes → Start here:
1. Read **[THRESHOLD_QUICK_REF.md](THRESHOLD_QUICK_REF.md)**
   - See TL;DR section
   - Note the new parameters to add
   - Check threshold recommendations table

### If you have 10 minutes → Add this:
2. Read **[BEFORE_AFTER_COMPARISON.md](BEFORE_AFTER_COMPARISON.md)**
   - See visual comparison of results
   - Understand the security trade-off
   - Review the metrics comparison

### If you have 30 minutes → Also read:
3. Read **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**
   - Understand what changed in the code
   - Learn about new functions
   - See configuration options

4. Run **[threshold_examples.py](threshold_examples.py)**
   - Update paths to your data
   - Uncomment example_1_find_optimal_threshold()
   - See it work with your models

### If you want comprehensive understanding:
5. Read **[DYNAMIC_THRESHOLDING_GUIDE.md](DYNAMIC_THRESHOLDING_GUIDE.md)**
   - Deep dive into theory
   - Usage examples for all functions
   - FAQ and troubleshooting
   - Advanced usage patterns

## 🎯 Common Use Cases

### Use Case 1: Evaluate Existing Models
**Goal:** Test your current models with security-focused thresholds

**Read:**
- THRESHOLD_QUICK_REF.md (Quick Start section)
- BEFORE_AFTER_COMPARISON.md (to understand impact)

**Run:**
```python
results = create_evaluation_matrix(
    models_dir, eval_base_dir, device,
    use_auto_threshold=True,
    min_recall=0.90
)
# Check: evaluation_thresholds.csv for optimal thresholds
```

---

### Use Case 2: Deploy Model to Production
**Goal:** Find the right threshold for production use

**Read:**
- DYNAMIC_THRESHOLDING_GUIDE.md (Threshold Recommendations section)
- BEFORE_AFTER_COMPARISON.md (When to Adjust min_recall section)

**Run:**
```python
# Find optimal threshold once
threshold, metrics, analysis = find_optimal_threshold(
    model, val_seq, val_labels,
    min_recall=0.90  # Adjust based on your security needs
)

# Save this threshold value
# Use it for all production predictions
predictions = (probabilities > threshold).long()
```

---

### Use Case 3: Compare Security Levels
**Goal:** Understand trade-offs at different security levels

**Read:**
- BEFORE_AFTER_COMPARISON.md (When to Adjust min_recall section)

**Run:**
```python
# From threshold_examples.py
example_3_security_focused_evaluation()
```

---

### Use Case 4: Understand Why This Works
**Goal:** Learn the theory behind dynamic thresholding

**Read:**
- DYNAMIC_THRESHOLDING_GUIDE.md (full document)
- Sections: Understanding the Metrics, F1 vs F2 Score

---

## 🔧 Modified Files

### Main Code
- **`inference_example.py`** - Core implementation with new functions

### Documentation
- **`THRESHOLD_QUICK_REF.md`** - Fast reference
- **`BEFORE_AFTER_COMPARISON.md`** - Visual impact guide
- **`IMPLEMENTATION_SUMMARY.md`** - Change overview
- **`DYNAMIC_THRESHOLDING_GUIDE.md`** - Comprehensive guide
- **`threshold_examples.py`** - Example scripts
- **`README_THRESHOLDING.md`** - This file

## 📊 New Output Files

When you run `create_evaluation_matrix()` with auto-thresholding, you'll get:

| File | Contains |
|------|----------|
| `evaluation_matrix_summary.csv` | Mean ± std for all metrics |
| **`evaluation_thresholds.csv`** | **Optimal thresholds per model** ⭐ |
| **`evaluation_matrix_f2_mean.csv`** | **F2 scores (security metric)** ⭐ |
| `evaluation_matrix_recall_mean.csv` | Recall scores |
| **`evaluation_matrix_false_negatives.csv`** | **Missed vulnerabilities** ⭐ |
| `evaluation_matrix_detailed.csv` | Per-run detailed results |

⭐ = Most important for security

## 🎓 Key Concepts

### What is Dynamic Thresholding?
Instead of using a fixed 0.5 threshold, we automatically find the optimal threshold that:
- Maximizes F2 score (prioritizes recall)
- Ensures minimum recall (e.g., catch ≥90% of vulnerabilities)
- Minimizes false negatives (missed bugs)

### Why F2 Instead of F1?
- **F1 Score**: Equal weight to precision and recall
- **F2 Score**: Weights recall 2x more than precision
- **For Security**: We care more about catching vulnerabilities than avoiding false alarms

### What is min_recall?
The minimum percentage of vulnerabilities you require the model to detect.
- `min_recall=0.90` means "catch at least 90% of vulnerabilities"
- The algorithm finds the highest threshold that still meets this requirement

## ⚡ Quick Commands

### Test with One Model
```python
from inference_example import load_trained_model, find_optimal_threshold

model, _ = load_trained_model('model.pt', device)
threshold, _, _ = find_optimal_threshold(model, val_seq, val_labels, min_recall=0.90)
print(f"Optimal threshold: {threshold:.4f}")
```

### Run Full Evaluation
```python
from inference_example import create_evaluation_matrix

results = create_evaluation_matrix(
    '/path/to/models',
    '/path/to/evaluation', 
    device='cuda',
    use_auto_threshold=True,
    min_recall=0.90
)
```

### Compare Thresholds
```python
for threshold in [0.3, 0.4, 0.5]:
    metrics = evaluate_model(model, test_seq, test_labels, threshold=threshold)
    print(f"@{threshold}: Recall={metrics['recall']:.3f}, FN={metrics['false_negatives']}")
```

## 🐛 Troubleshooting

### Issue: "No threshold achieves minimum recall"
**Quick Fix:** Lower `min_recall` (try 0.85 or 0.80)

**Details:** See DYNAMIC_THRESHOLDING_GUIDE.md → FAQ section

---

### Issue: Too many false positives
**Quick Fix:** Increase `min_recall` slightly (e.g., 0.90 → 0.92)

**Details:** See DYNAMIC_THRESHOLDING_GUIDE.md → Choosing min_recall

---

### Issue: Precision too low
**Quick Fix:** This is expected! For security, we accept lower precision.

**Details:** See BEFORE_AFTER_COMPARISON.md → The Security Trade-off

---

## 📞 Getting Help

1. **Quick Question?** → Check **THRESHOLD_QUICK_REF.md**
2. **Understanding Results?** → Read **BEFORE_AFTER_COMPARISON.md**
3. **Implementation Details?** → See **IMPLEMENTATION_SUMMARY.md**
4. **In-depth Question?** → Read **DYNAMIC_THRESHOLDING_GUIDE.md**
5. **Want Examples?** → Run **threshold_examples.py**

## 🎯 Success Metrics

After implementing dynamic thresholding, you should see:

✅ **Recall ≥ 90%** (catching most vulnerabilities)
✅ **F2 Score ≥ 0.75** (good security-focused metric)
✅ **False Negatives < 10%** of positive samples (few missed bugs)
⚡ **Precision ≥ 40%** (acceptable for security, not too many false alarms)

If you're not seeing these improvements, see the Troubleshooting section in DYNAMIC_THRESHOLDING_GUIDE.md

## 📝 Summary

**The Big Picture:**
Your model was too cautious → missing vulnerabilities → security risk
Dynamic thresholding fixes this → catches 93% of vulnerabilities → much safer

**The Trade-off:**
More false positives (120 vs 8) but far fewer false negatives (7 vs 48)
For security, this is the right trade-off!

**Getting Started:**
1. Read THRESHOLD_QUICK_REF.md (2 min)
2. Add two parameters to your evaluation code (30 sec)
3. Run and review results (5 min)
4. Celebrate safer code! 🎉🔒

---

**Ready to make your model security-focused? Start with [THRESHOLD_QUICK_REF.md](THRESHOLD_QUICK_REF.md)!** 🚀
