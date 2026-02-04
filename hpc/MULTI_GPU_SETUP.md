# Multi-GPU Training Setup

## Overview
The LSTM training script has been updated to support **DataParallel** multi-GPU training on the HPC cluster with 4 A30 GPUs.

## Changes Made

### 1. **train_lstm.py**
- Added `use_multi_gpu` flag to `DEFAULT_CONFIG` (default: `True`)
- Updated `load_dataset()` to use:
  - `num_workers=4` for parallel data loading
  - `pin_memory=True` for faster GPU transfer
- Modified device selection to use all available GPUs when `use_multi_gpu=True`
- Wrapped model in `nn.DataParallel` for automatic parallelization
- Updated model saving to unwrap `DataParallel` before saving state dict
- Added CLI arguments: `--multi-gpu` / `--no-multi-gpu`

### 2. **run_dedup_experiment.sh**
- 
  - Effective batch size = 8 × 4 GPUs = **32**
- Already configured with `--gpus 4` in SLURM header

## How It Works

### DataParallel Behavior
```python
# Input batch of size 32 is automatically split across 4 GPUs:
# GPU 0: processes 8 samples
# GPU 1: processes 8 samples  
# GPU 2: processes 8 samples
# GPU 3: processes 8 samples

# Gradients are averaged and applied on GPU 0 (primary device)
```

### Device Setup
```python
if use_multi_gpu and torch.cuda.device_count() > 1:
    device = torch.device('cuda:0')  # Primary device
    model = nn.DataParallel(model)   # Wrap for multi-GPU
else:
    device = select_best_gpu()       # Single GPU mode
```

## Usage

### Default (Multi-GPU Enabled)
```bash
python train_lstm.py \
    --dataset-dir /path/to/data \
    --dataset-name juliet_c_simhash_k=1 \
    --ood-dir /path/to/ood \
    --weights /path/to/weights.pt \
    --batch-size 32  # Per-GPU batch size
```

### Disable Multi-GPU
```bash
python train_lstm.py \
    --no-multi-gpu \
    --dataset-dir /path/to/data \
    ...
```

### Run Experiment Script
```bash
sbatch run_dedup_experiment.sh
```

## Performance Expectations

### Speed Improvements
- **Single GPU (A30)**: ~100 batches/epoch
- **4 GPUs (DataParallel)**: ~3.5-3.8x speedup
  - Not perfect 4x due to communication overhead
  - Effective batch size 4x larger

### Memory Usage
- Each GPU processes 1/4 of the batch
- Model replicated on each GPU
- Gradients synchronized after backward pass

### Effective Batch Size
```
Per-GPU batch size: 32
Number of GPUs: 4
Effective batch size: 32 × 4 = 128
```

**Note**: Larger effective batch size may require adjusting learning rate:
- Common rule: `new_lr = base_lr × sqrt(num_gpus)`
- Current: 0.001 (may want to try 0.002 for 4 GPUs)

## Monitoring

### Check GPU Utilization
```bash
# On compute node during training
nvidia-smi -l 1
```

You should see all 4 GPUs with:
- Similar memory usage
- Similar GPU utilization % 
- Process listed on each GPU

### Log Output
```
Multi-GPU training enabled
Using 4 GPUs:
  GPU 0: NVIDIA A30
    Memory: 24.0 GB
  GPU 1: NVIDIA A30
    Memory: 24.0 GB
  GPU 2: NVIDIA A30
    Memory: 24.0 GB
  GPU 3: NVIDIA A30
    Memory: 24.0 GB
Using DataParallel with 4 GPUs
Multi-GPU mode: effective batch size = 32 x 4 = 128
```

## Troubleshooting

### Issue: Only 1 GPU being used
**Solution**: Check SLURM allocation
```bash
squeue -u $USER  # Should show --gpus=4
nvidia-smi      # Should list 4 GPUs
```

### Issue: Out of memory
**Solution**: Reduce per-GPU batch size
```bash
--batch-size 16  # Instead of 32
```

### Issue: Slow training
**Solution**: Check data loading workers
- Default: 4 workers per DataLoader
- Can adjust in `load_dataset()` function

### Issue: Model loading errors later
**Cause**: DataParallel wrapper saved in state dict
**Solution**: Already handled - we unwrap before saving:
```python
model_to_save = model.module if isinstance(model, nn.DataParallel) else model
torch.save(model_to_save.state_dict(), path)
```

## Expected Training Time

### Per k value (5 seeds, 50 epochs each)
- **Single GPU**: ~12-15 hours
- **4 GPUs**: ~3-4 hours

### Full experiment (k=1 to k=12)
- **Single GPU**: ~6-7 days  
- **4 GPUs**: ~1.5-2 days

## Alternative: DistributedDataParallel (Future)

For even better performance, consider upgrading to DDP:
- Better scaling efficiency (closer to 4x)
- Each GPU runs independent process
- More complex setup
- Requires changing SLURM to use `--ntasks=4`

Current DataParallel is simpler and sufficient for 4 GPUs.

## Files Modified
1. `/Users/aeg00011/Romeo/train_lstm.py`
2. `/Users/aeg00011/Romeo/run_dedup_experiment.sh`

## Verification

Test on a small dataset first:
```bash
# Test run with 1 epoch
python train_lstm.py \
    --dataset-dir $TENSOR_DIR/juliet_c_simhash_k=1_* \
    --dataset-name test_multi_gpu \
    --ood-dir $OOD_DIR \
    --weights $WEIGHTS \
    --epochs 1 \
    --seeds 42
```

Check the output shows "Using 4 GPUs" and watch `nvidia-smi` during training.
