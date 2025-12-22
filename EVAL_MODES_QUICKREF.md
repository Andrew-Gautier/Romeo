# Quick Reference: Evaluation Modes

## Commands

| Mode | Command | Output |
|------|---------|--------|
| **Split** (default) | `python pretraining_generation.py` | train/val/test folders |
| **Split** (explicit) | `python pretraining_generation.py --eval-mode split` | train/val/test folders |
| **Full** | `python pretraining_generation.py --eval-mode full` | single 'full' folder |

## Output Structure

### Split Mode
```
evaluation/c/
├── train/       (60%)
├── validation/  (20%)
└── test/        (20%)
```

### Full Mode
```
evaluation/c/
└── full/        (100%)
```

## Use Cases

| Scenario | Mode | Reason |
|----------|------|--------|
| Fine-tune on eval data | Split | Need train/val separation |
| Benchmark pretrained models | Full | No training, just inference |
| Hyperparameter tuning | Split | Need validation set |
| Cross-validation | Full | Custom splitting |
| Maximum eval data | Full | All samples available |
| Traditional ML workflow | Split | Standard train/val/test |

## Quick Examples

```bash
# Default: split mode with 10k samples
python pretraining_generation.py

# Full dataset for benchmarking
python pretraining_generation.py --eval-mode full

# Small test dataset
python pretraining_generation.py --pretraining-samples 1000 --eval-samples 1000

# Large full dataset
python pretraining_generation.py --eval-mode full --eval-samples 50000
```

## Loading Data

### Split Mode
```python
test_seq = torch.load('tensors/TIMESTAMP/evaluation/c/test/sequences.pt')
test_lab = torch.load('tensors/TIMESTAMP/evaluation/c/test/labels.pt')
```

### Full Mode
```python
full_seq = torch.load('tensors/TIMESTAMP/evaluation/c/full/sequences.pt')
full_lab = torch.load('tensors/TIMESTAMP/evaluation/c/full/labels.pt')
```

## Check Mode from Metadata

```python
import json
with open('tensors/TIMESTAMP/pipeline_metadata.json') as f:
    mode = json.load(f)['eval_mode']  # 'split' or 'full'
```
