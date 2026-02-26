"""
LoRA Fine-Tuning Script for Vulnerability Detection

Fine-tunes full pretrained transformer models (DeepSeek-Coder, CodeLlama) with
LoRA adapters for binary vulnerability classification on the Devign dataset.

This is fundamentally different from the LSTM experiments:
  - LSTM Experiments: Use only the embedding layer from pretrained models,
    fed into a separately-trained LSTM+Attention classifier.
  - LoRA Experiments: Use the FULL pretrained transformer, adding small
    trainable LoRA adapters to the attention layers. The transformer's
    contextual representations are used directly for classification.

Architecture:
  Source code → Model-specific Tokenizer → Full Transformer + LoRA → Classification Head

Models supported:
  - deepseek-coder  (deepseek-ai/deepseek-coder-6.7b-base)
  - codellama       (codellama/CodeLlama-7b-hf)

Requirements:
  pip install torch transformers peft accelerate bitsandbytes scikit-learn tqdm

Usage (single GPU):
  python train_lora.py \
      --model-name deepseek-coder \
      --dataset-dir /path/to/devign_tensors \
      --ood-dir /path/to/juliet_ood_tensors \
      --output-dir /path/to/output \
      --hf-cache-dir /path/to/huggingface

Usage (multi-GPU with accelerate):
  accelerate launch train_lora.py \
      --model-name deepseek-coder \
      --dataset-dir /path/to/devign_tensors \
      --ood-dir /path/to/juliet_ood_tensors \
      --output-dir /path/to/output \
      --hf-cache-dir /path/to/huggingface
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os
import json
import csv
import time
import argparse
import glob
import re
from datetime import datetime, timedelta
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)


# ============================================================================
# Configuration
# ============================================================================

HF_MODEL_CONFIGS = {
    'deepseek-coder': {
        'model_id': 'deepseek-ai/deepseek-coder-6.7b-base',
        'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        'max_seq_length': 4096,
    },
    'codellama': {
        'model_id': 'codellama/CodeLlama-7b-hf',
        'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        'max_seq_length': 4096,
    },
}

DEFAULT_LORA_CONFIG = {
    'r': 16,               # LoRA rank
    'lora_alpha': 32,      # LoRA scaling factor
    'lora_dropout': 0.1,   # Dropout for LoRA layers
    'bias': 'none',        # Don't train bias terms
}

DEFAULT_TRAINING_CONFIG = {
    'batch_size': 4,        # Per-GPU batch size (LoRA models are much larger)
    'gradient_accumulation_steps': 8,  # Effective batch = 4 * 8 = 32
    'learning_rate': 2e-4,  # Standard LoRA learning rate
    'weight_decay': 0.01,
    'epochs': 10,           # LoRA converges faster than training from scratch
    'patience': 3,
    'warmup_ratio': 0.1,    # 10% of steps for warmup
    'max_grad_norm': 1.0,
    'fp16': True,           # Use mixed precision
}

SEEDS = list(range(1, 21))


# ============================================================================
# Data Loading (from pre-tokenized tensors)
# ============================================================================

def load_tensor_dataset(data_dir, split='train', batch_size=4, shuffle=True):
    """
    Load pre-tokenized tensor dataset.
    
    NOTE: The tensors must have been tokenized with the SAME tokenizer as the
    model being fine-tuned. Using mismatched tokenizers will produce garbage.
    
    Args:
        data_dir: Path to directory with {split}_sequences.pt, {split}_labels.pt
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        shuffle: Whether to shuffle
    
    Returns:
        DataLoader
    """
    seq_path = os.path.join(data_dir, f'{split}_sequences.pt')
    label_path = os.path.join(data_dir, f'{split}_labels.pt')
    
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"Not found: {seq_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Not found: {label_path}")
    
    sequences = torch.load(seq_path, map_location='cpu').long()
    labels = torch.load(label_path, map_location='cpu').long()
    
    print(f"  {split}: {sequences.shape[0]} samples, seq_len={sequences.shape[1]}")
    
    # Create attention mask (1 for real tokens, 0 for padding)
    # Assumes padding value is 0
    attention_mask = (sequences != 0).long()
    
    dataset = torch.utils.data.TensorDataset(sequences, attention_mask, labels)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=2,
        drop_last=(split == 'train'),
    )
    
    return loader


# ============================================================================
# Model Creation
# ============================================================================

def create_lora_model(model_name, hf_cache_dir, lora_config_dict, device, fp16=True):
    """
    Load a pretrained causal LM and wrap it for sequence classification with LoRA.
    
    Args:
        model_name: Key in HF_MODEL_CONFIGS ('deepseek-coder' or 'codellama')
        hf_cache_dir: Path to HuggingFace cache directory
        lora_config_dict: LoRA hyperparameters dict
        device: torch device
        fp16: Whether to use float16
    
    Returns:
        tuple: (model, tokenizer)
    """
    if model_name not in HF_MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Options: {list(HF_MODEL_CONFIGS.keys())}")
    
    model_id = HF_MODEL_CONFIGS[model_name]['model_id']
    target_modules = HF_MODEL_CONFIGS[model_name]['target_modules']
    
    print(f"Loading model: {model_id}")
    print(f"  Cache dir: {hf_cache_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=hf_cache_dir,
        trust_remote_code=True,
    )
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token = eos_token: '{tokenizer.pad_token}'")
    
    # Load model for sequence classification (binary)
    # This adds a classification head on top of the base model
    model_kwargs = {
        'cache_dir': hf_cache_dir,
        'num_labels': 2,
        'trust_remote_code': True,
        'torch_dtype': torch.float16 if fp16 else torch.float32,
    }
    
    # Try to load with device_map for large models
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            **model_kwargs,
        )
    except Exception as e:
        print(f"  Standard loading failed ({e}), trying with device_map='auto'...")
        model_kwargs['device_map'] = 'auto'
        model_kwargs['load_in_8bit'] = False
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            **model_kwargs,
        )
    
    # Set pad token id in model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_config_dict['r'],
        lora_alpha=lora_config_dict['lora_alpha'],
        lora_dropout=lora_config_dict['lora_dropout'],
        bias=lora_config_dict['bias'],
        target_modules=target_modules,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print parameter counts
    model.print_trainable_parameters()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.4f}%")
    
    return model, tokenizer


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, device, 
                max_grad_norm=1.0, gradient_accumulation_steps=1, fp16=True):
    """Train for one epoch with gradient accumulation and optional mixed precision."""
    model.train()
    total_loss = 0
    num_steps = 0
    
    scaler = torch.amp.GradScaler('cuda') if fp16 and device.type == 'cuda' else None
    
    optimizer.zero_grad()
    
    progress = tqdm(train_loader, desc='Training', leave=False)
    for step, (input_ids, attention_mask, labels) in enumerate(progress):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        if fp16 and scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_steps += 1
        
        if (step + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        progress.set_postfix({'loss': f'{total_loss / num_steps:.4f}'})
    
    # Handle remaining gradients
    if len(train_loader) % gradient_accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return total_loss / num_steps


def evaluate(model, data_loader, device, fp16=True):
    """Evaluate model and return metrics."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(data_loader, desc='Evaluating', leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            if fp16 and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs.loss.item()
            
            # Get probabilities for the positive class
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            all_predictions.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()
    binary_preds = (all_predictions >= 0.5).astype(float)
    
    try:
        auroc = roc_auc_score(all_labels, all_predictions)
    except ValueError:
        auroc = 0.5
    
    results = {
        'loss': total_loss / len(data_loader),
        'auroc': float(auroc),
        'accuracy': float(accuracy_score(all_labels, binary_preds)),
        'precision': float(precision_score(all_labels, binary_preds, zero_division=0)),
        'recall': float(recall_score(all_labels, binary_preds, zero_division=0)),
        'f1': float(f1_score(all_labels, binary_preds, zero_division=0)),
    }
    
    return results


# ============================================================================
# Full Training Loop
# ============================================================================

def train_model_with_seed(
    model_name, hf_cache_dir, train_loader, val_loader,
    lora_config_dict, training_config, device, seed, checkpoint_dir
):
    """
    Train a LoRA model for one seed.
    
    Returns:
        tuple: (model, training_history)
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create model
    model, tokenizer = create_lora_model(
        model_name, hf_cache_dir, lora_config_dict, device,
        fp16=training_config['fp16']
    )
    model = model.to(device)
    
    # Optimizer — only optimize trainable (LoRA) parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
    )
    
    # Scheduler
    total_steps = (
        len(train_loader) // training_config['gradient_accumulation_steps'] 
        * training_config['epochs']
    )
    warmup_steps = int(total_steps * training_config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auroc': [],
        'val_f1': [],
        'train_times': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
    }
    
    best_model_path = None
    epochs_since_improvement = 0
    
    print(f"\n{'='*60}")
    print(f"Training with seed {seed}")
    print(f"  Total steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"  Effective batch size: {training_config['batch_size'] * training_config['gradient_accumulation_steps']}")
    print(f"{'='*60}")
    
    total_start = time.time()
    
    for epoch in range(training_config['epochs']):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            max_grad_norm=training_config['max_grad_norm'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            fp16=training_config['fp16'],
        )
        
        # Validate
        val_results = evaluate(model, val_loader, device, fp16=training_config['fp16'])
        
        epoch_time = time.time() - epoch_start
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_results['loss'])
        history['val_auroc'].append(val_results['auroc'])
        history['val_f1'].append(val_results['f1'])
        history['train_times'].append(epoch_time)
        
        print(f"  Epoch {epoch+1}/{training_config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_results['loss']:.4f} | "
              f"Val AUROC: {val_results['auroc']:.4f} | "
              f"Val F1: {val_results['f1']:.4f} | "
              f"Time: {timedelta(seconds=int(epoch_time))}")
        
        # Early stopping
        if val_results['loss'] < history['best_val_loss']:
            history['best_val_loss'] = val_results['loss']
            history['best_epoch'] = epoch + 1
            epochs_since_improvement = 0
            
            # Save best LoRA adapter weights
            best_model_path = os.path.join(checkpoint_dir, f'seed{seed}_best_lora')
            model.save_pretrained(best_model_path)
            print(f"    ✓ New best model saved to: {best_model_path}")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= training_config['patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    history['total_time'] = time.time() - total_start
    
    # Reload best model
    if best_model_path and os.path.exists(best_model_path):
        print(f"  Loading best model from epoch {history['best_epoch']}...")
        # Re-create base model and load best LoRA weights
        base_model = AutoModelForSequenceClassification.from_pretrained(
            HF_MODEL_CONFIGS[model_name]['model_id'],
            cache_dir=hf_cache_dir,
            num_labels=2,
            torch_dtype=torch.float16 if training_config['fp16'] else torch.float32,
            trust_remote_code=True,
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base_model, best_model_path)
        model = model.to(device)
    
    return model, tokenizer, history


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(
    model_name, hf_cache_dir, dataset_dir, ood_dir, output_dir,
    lora_config_dict, training_config, seeds, device
):
    """
    Run the full LoRA fine-tuning experiment across multiple seeds.
    
    For each seed:
      1. Create model with LoRA adapters
      2. Train on train split with early stopping on val split
      3. Evaluate on test split and OOD split
      4. Save LoRA adapter weights and results
    """
    print(f"\n{'#'*70}")
    print(f"# LoRA Experiment: {model_name}")
    print(f"# Dataset: {dataset_dir}")
    print(f"# OOD: {ood_dir}")
    print(f"{'#'*70}")
    
    # Create directories
    experiment_dir = os.path.join(output_dir, f'lora_{model_name}')
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    results_dir = os.path.join(experiment_dir, 'results')
    plots_dir = os.path.join(experiment_dir, 'plots')
    for d in [checkpoint_dir, results_dir, plots_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Load data
    print(f"\nLoading datasets...")
    train_loader = load_tensor_dataset(
        dataset_dir, 'train', training_config['batch_size'], shuffle=True
    )
    val_loader = load_tensor_dataset(
        dataset_dir, 'val', training_config['batch_size'], shuffle=False
    )
    test_loader = load_tensor_dataset(
        dataset_dir, 'test', training_config['batch_size'], shuffle=False
    )
    ood_loader = load_tensor_dataset(
        ood_dir, 'test', training_config['batch_size'], shuffle=False
    )
    
    # Results storage
    all_results = {
        'model': model_name,
        'model_id': HF_MODEL_CONFIGS[model_name]['model_id'],
        'dataset_dir': dataset_dir,
        'ood_dir': ood_dir,
        'lora_config': lora_config_dict,
        'training_config': training_config,
        'timestamp': datetime.now().isoformat(),
        'seeds': {},
    }
    
    # Train with each seed
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        
        # Check if already completed
        result_path = os.path.join(results_dir, f'seed{seed}_results.json')
        if os.path.exists(result_path):
            print(f"Seed {seed} already completed, loading existing results...")
            with open(result_path) as f:
                all_results['seeds'][seed] = json.load(f)
            continue
        
        try:
            # Train
            model, tokenizer, history = train_model_with_seed(
                model_name, hf_cache_dir, train_loader, val_loader,
                lora_config_dict, training_config, device, seed, checkpoint_dir
            )
            
            # Evaluate on test set
            print("\nEvaluating on test set...")
            test_results = evaluate(model, test_loader, device, fp16=training_config['fp16'])
            print(f"  Test: AUROC={test_results['auroc']:.4f}, F1={test_results['f1']:.4f}, "
                  f"Acc={test_results['accuracy']:.4f}")
            
            # Evaluate on OOD set
            print("Evaluating on OOD set...")
            ood_results = evaluate(model, ood_loader, device, fp16=training_config['fp16'])
            print(f"  OOD:  AUROC={ood_results['auroc']:.4f}, F1={ood_results['f1']:.4f}, "
                  f"Acc={ood_results['accuracy']:.4f}")
            
            # Store results for this seed
            seed_results = {
                'history': history,
                'test': test_results,
                'ood': ood_results,
            }
            all_results['seeds'][seed] = seed_results
            
            # Save individual seed results
            with open(result_path, 'w') as f:
                json.dump(seed_results, f, indent=2, default=str)
            
            # Plot training curves
            plot_training_curves(history, seed, plots_dir)
            
        except Exception as e:
            print(f"ERROR on seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            all_results['seeds'][seed] = {'error': str(e)}
        
        finally:
            # Free GPU memory
            if 'model' in dir():
                del model
            torch.cuda.empty_cache()
    
    # Aggregate results
    successful_seeds = {
        k: v for k, v in all_results['seeds'].items()
        if 'error' not in v and 'test' in v
    }
    
    if successful_seeds:
        all_results['aggregate'] = compute_aggregate_stats(successful_seeds)
        all_results['completed_seeds'] = sorted(successful_seeds.keys())
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETE: {model_name}")
        print(f"{'='*60}")
        print(f"Completed seeds: {len(successful_seeds)}/{len(seeds)}")
        agg = all_results['aggregate']
        print(f"  Test AUROC: {agg['test_auroc_mean']:.4f} ± {agg['test_auroc_std']:.4f}")
        print(f"  Test F1:    {agg['test_f1_mean']:.4f} ± {agg['test_f1_std']:.4f}")
        print(f"  OOD AUROC:  {agg['ood_auroc_mean']:.4f} ± {agg['ood_auroc_std']:.4f}")
        print(f"  OOD F1:     {agg['ood_f1_mean']:.4f} ± {agg['ood_f1_std']:.4f}")
    
    # Save full results
    results_path = os.path.join(results_dir, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    # Save CSV summaries
    save_csv_summaries(all_results, results_dir)
    
    return all_results


def compute_aggregate_stats(seeds_results):
    """Compute mean/std across seeds for all metrics."""
    metrics = ['auroc', 'f1', 'accuracy', 'precision', 'recall', 'loss']
    aggregate = {}
    
    for split in ['test', 'ood']:
        for metric in metrics:
            values = [s[split][metric] for s in seeds_results.values() if split in s]
            if values:
                aggregate[f'{split}_{metric}_mean'] = float(np.mean(values))
                aggregate[f'{split}_{metric}_std'] = float(np.std(values))
    
    return aggregate


def save_csv_summaries(all_results, results_dir):
    """Save per-seed and summary CSVs."""
    # Per-seed CSV
    csv_path = os.path.join(results_dir, 'per_seed_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'split', 'auroc', 'f1', 'accuracy', 'precision', 'recall', 'loss'])
        for seed, data in sorted(all_results['seeds'].items()):
            if 'error' in data:
                continue
            for split in ['test', 'ood']:
                if split in data:
                    m = data[split]
                    writer.writerow([
                        seed, split,
                        f"{m['auroc']:.6f}", f"{m['f1']:.6f}",
                        f"{m['accuracy']:.6f}", f"{m['precision']:.6f}",
                        f"{m['recall']:.6f}", f"{m['loss']:.6f}",
                    ])
    print(f"Per-seed CSV saved to: {csv_path}")
    
    # Summary CSV
    if 'aggregate' in all_results:
        summary_path = os.path.join(results_dir, 'results_summary.csv')
        agg = all_results['aggregate']
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'model', 'test_auroc_mean', 'test_auroc_std',
                'test_f1_mean', 'test_f1_std',
                'ood_auroc_mean', 'ood_auroc_std',
                'ood_f1_mean', 'ood_f1_std',
            ])
            writer.writerow([
                all_results['model'],
                f"{agg.get('test_auroc_mean', 0):.6f}", f"{agg.get('test_auroc_std', 0):.6f}",
                f"{agg.get('test_f1_mean', 0):.6f}", f"{agg.get('test_f1_std', 0):.6f}",
                f"{agg.get('ood_auroc_mean', 0):.6f}", f"{agg.get('ood_auroc_std', 0):.6f}",
                f"{agg.get('ood_f1_mean', 0):.6f}", f"{agg.get('ood_f1_std', 0):.6f}",
            ])
        print(f"Summary CSV saved to: {summary_path}")


def plot_training_curves(history, seed, plots_dir):
    """Plot and save training curves for a single seed."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'], label='Validation')
    axes[0].axvline(x=history['best_epoch'], color='r', linestyle='--', alpha=0.5, label='Best')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Loss (Seed {seed})')
    axes[0].legend()
    
    # AUROC
    axes[1].plot(epochs, history['val_auroc'], label='Val AUROC', color='green')
    axes[1].axvline(x=history['best_epoch'], color='r', linestyle='--', alpha=0.5, label='Best')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUROC')
    axes[1].set_title(f'Validation AUROC (Seed {seed})')
    axes[1].set_ylim(0.4, 1.0)
    axes[1].legend()
    
    # F1
    axes[2].plot(epochs, history['val_f1'], label='Val F1', color='orange')
    axes[2].axvline(x=history['best_epoch'], color='r', linestyle='--', alpha=0.5, label='Best')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1')
    axes[2].set_title(f'Validation F1 (Seed {seed})')
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'training_seed{seed}.png'), dpi=150)
    plt.close()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LoRA Fine-Tuning for Vulnerability Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument('--model-name', type=str, required=True,
                        choices=list(HF_MODEL_CONFIGS.keys()),
                        help='Model to fine-tune')
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to training dataset (with train/val/test splits)')
    parser.add_argument('--ood-dir', type=str, required=True,
                        help='Path to OOD evaluation dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--hf-cache-dir', type=str, required=True,
                        help='Path to HuggingFace cache directory')
    
    # LoRA hyperparameters
    parser.add_argument('--lora-r', type=int, default=DEFAULT_LORA_CONFIG['r'],
                        help=f"LoRA rank (default: {DEFAULT_LORA_CONFIG['r']})")
    parser.add_argument('--lora-alpha', type=int, default=DEFAULT_LORA_CONFIG['lora_alpha'],
                        help=f"LoRA alpha (default: {DEFAULT_LORA_CONFIG['lora_alpha']})")
    parser.add_argument('--lora-dropout', type=float, default=DEFAULT_LORA_CONFIG['lora_dropout'],
                        help=f"LoRA dropout (default: {DEFAULT_LORA_CONFIG['lora_dropout']})")
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=DEFAULT_TRAINING_CONFIG['batch_size'],
                        help=f"Batch size per GPU (default: {DEFAULT_TRAINING_CONFIG['batch_size']})")
    parser.add_argument('--grad-accum', type=int, 
                        default=DEFAULT_TRAINING_CONFIG['gradient_accumulation_steps'],
                        help=f"Gradient accumulation steps (default: {DEFAULT_TRAINING_CONFIG['gradient_accumulation_steps']})")
    parser.add_argument('--lr', type=float, default=DEFAULT_TRAINING_CONFIG['learning_rate'],
                        help=f"Learning rate (default: {DEFAULT_TRAINING_CONFIG['learning_rate']})")
    parser.add_argument('--epochs', type=int, default=DEFAULT_TRAINING_CONFIG['epochs'],
                        help=f"Max epochs (default: {DEFAULT_TRAINING_CONFIG['epochs']})")
    parser.add_argument('--patience', type=int, default=DEFAULT_TRAINING_CONFIG['patience'],
                        help=f"Early stopping patience (default: {DEFAULT_TRAINING_CONFIG['patience']})")
    parser.add_argument('--no-fp16', action='store_true',
                        help='Disable mixed precision training')
    
    # Seeds
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                        help='Random seeds (default: 1-20)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    # Build configs
    lora_config_dict = {
        'r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'bias': 'none',
    }
    
    training_config = {
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.grad_accum,
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'epochs': args.epochs,
        'patience': args.patience,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        'fp16': not args.no_fp16,
    }
    
    print(f"\nLoRA Config: {json.dumps(lora_config_dict, indent=2)}")
    print(f"Training Config: {json.dumps(training_config, indent=2)}")
    
    # Run experiment
    run_experiment(
        model_name=args.model_name,
        hf_cache_dir=args.hf_cache_dir,
        dataset_dir=args.dataset_dir,
        ood_dir=args.ood_dir,
        output_dir=args.output_dir,
        lora_config_dict=lora_config_dict,
        training_config=training_config,
        seeds=args.seeds,
        device=device,
    )


if __name__ == '__main__':
    main()
