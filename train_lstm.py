"""
LSTM Training Script for Deduplication Experiment
Trains models on Juliet C SimHash datasets (k=1 to k=12) with multiple seeds.
Evaluates on out-of-distribution (OOD) dataset (Devign).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
import os
import json
import time
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
from hpc.gpu_utils import select_best_gpu, clear_gpu_memory

from binary_classifier import LSTMClassifier, create_model


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'lstm_nodes': 256,
    'vocab_size': 49152,
    'embedding_size': 4096,
    'output_dim': 1,
    'patience': 5,
    'gradient_clip': 1.0,
    'n_layers': 2,
    'bidirectional': True,
    'dropout': 0.5,
    'n_heads': 8,  # Number of attention heads (must divide lstm_nodes * 2 evenly)
    'use_multi_gpu': True,  # Enable DataParallel for multi-GPU training
}

SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds for reproducibility


# ============================================================================
# Data Loading
# ============================================================================

def load_dataset(data_dir, split='train', batch_size=32, shuffle=True, num_workers=0):
    """
    Load a dataset split from a directory.
    
    Args:
        data_dir (str): Path to dataset directory
        split (str): One of 'train', 'val', 'test'
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading.
                          Default is 0 (main process only) for CUDA compatibility.
                          Using num_workers > 0 with CUDA can cause deadlocks.
        
    Returns:
        DataLoader: PyTorch DataLoader for the split
    """
    sequences = torch.load(os.path.join(data_dir, f'{split}_sequences.pt')).long()
    labels = torch.load(os.path.join(data_dir, f'{split}_labels.pt'))
    
    dataset = TensorDataset(sequences, labels)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=(split == 'train'),
        num_workers=num_workers,
        pin_memory=(num_workers == 0)  # Only use pin_memory when num_workers=0
    )
    
    return loader


def load_cwe_indices(data_dir, split='test'):
    """Load CWE indices for per-CWE evaluation."""
    cwe_path = os.path.join(data_dir, f'{split}_cwe_indices.pt')
    if os.path.exists(cwe_path):
        return torch.load(cwe_path)
    return None


def load_cwe_mapping(data_dir):
    """Load CWE to index mapping."""
    mapping_path = os.path.join(data_dir, 'idx_to_cwe.pt')
    if os.path.exists(mapping_path):
        return torch.load(mapping_path, weights_only=False)
    return None


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, iterator, optimizer, criterion, device, gradient_clip=1.0):
    """Train for one epoch."""
    epoch_loss = 0
    model.train()
    
    for batch_sequences, batch_labels in tqdm(iterator, desc='Training', leave=False):
        batch_sequences = batch_sequences.to(device)
        batch_labels = batch_labels.to(device).float()
        
        optimizer.zero_grad()
        predictions = model(batch_sequences).squeeze(1)
        
        loss = criterion(predictions, batch_labels)
        loss.backward()
        
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    """Evaluate model on a dataset."""
    epoch_loss = 0
    model.eval()
    
    # Initialize metrics
    metrics = {
        'auroc': BinaryAUROC().to(device),
        'accuracy': BinaryAccuracy().to(device),
        'precision': BinaryPrecision().to(device),
        'recall': BinaryRecall().to(device),
        'f1': BinaryF1Score().to(device),
    }
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_sequences, batch_labels in tqdm(iterator, desc='Evaluating', leave=False):
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device).float()
            
            predictions = model(batch_sequences).squeeze(1)
            
            # Update metrics
            for metric in metrics.values():
                metric.update(predictions, batch_labels.int())
            
            loss = criterion(predictions, batch_labels)
            epoch_loss += loss.item()
            
            all_predictions.append(predictions.cpu())
            all_labels.append(batch_labels.cpu())
    
    # Compute final metrics
    results = {
        'loss': epoch_loss / len(iterator),
        'auroc': metrics['auroc'].compute().item(),
        'accuracy': metrics['accuracy'].compute().item(),
        'precision': metrics['precision'].compute().item(),
        'recall': metrics['recall'].compute().item(),
        'f1': metrics['f1'].compute().item(),
    }
    
    # Reset metrics
    for metric in metrics.values():
        metric.reset()
    
    return results, torch.cat(all_predictions), torch.cat(all_labels)


def evaluate_per_cwe(predictions, labels, cwe_indices, idx_to_cwe, device):
    """Evaluate metrics per CWE category."""
    if cwe_indices is None or idx_to_cwe is None:
        return {}
    
    cwe_results = {}
    unique_cwes = torch.unique(cwe_indices)
    
    auroc_metric = BinaryAUROC().to(device)
    
    for cwe_idx in unique_cwes:
        cwe_idx = cwe_idx.item()
        mask = cwe_indices == cwe_idx
        
        if mask.sum() < 2:  # Need at least 2 samples
            continue
        
        cwe_preds = predictions[mask].to(device)
        cwe_labels = labels[mask].to(device)
        
        # Check if we have both classes
        if len(torch.unique(cwe_labels)) < 2:
            continue
        
        auroc_metric.reset()
        auroc_metric.update(cwe_preds, cwe_labels.int())
        
        cwe_name = idx_to_cwe.get(cwe_idx, f'CWE_{cwe_idx}')
        cwe_results[cwe_name] = {
            'auroc': auroc_metric.compute().item(),
            'n_samples': mask.sum().item(),
            'n_positive': cwe_labels.sum().item(),
        }
    
    return cwe_results


# ============================================================================
# Main Training Loop
# ============================================================================

def train_model(
    train_loader, 
    val_loader, 
    config, 
    pretrained_weights,
    device,
    checkpoint_dir,
    seed
):
    """
    Train a model with early stopping.
    
    Returns:
        tuple: (model, training_history)
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create model
    model = create_model(
        {
            'vocab_size': config['vocab_size'],
            'embedding_dim': config['embedding_size'],
            'hidden_dim': config['lstm_nodes'],
            'output_dim': config['output_dim'],
            'n_layers': config['n_layers'],
            'bidirectional': config['bidirectional'],
            'dropout': config['dropout'],
            'n_heads': config['n_heads'],
        },
        pretrained_weights=pretrained_weights,
        device=device
    )
    
    # Wrap model in DataParallel for multi-GPU training
    if config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCELoss().to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auroc': [],
        'train_times': [],
        'eval_times': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
    }
    
    best_model_state = None
    epochs_since_improvement = 0
    
    print(f"\n{'='*60}")
    print(f"Training with seed {seed}")
    print(f"{'='*60}")
    
    total_start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Train
        train_start = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            device, config['gradient_clip']
        )
        train_time = time.time() - train_start
        
        # Validate
        eval_start = time.time()
        val_results, _, _ = evaluate(model, val_loader, criterion, device)
        eval_time = time.time() - eval_start
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_results['loss'])
        history['val_auroc'].append(val_results['auroc'])
        history['train_times'].append(train_time)
        history['eval_times'].append(eval_time)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_results['loss']:.4f} | "
              f"Val AUROC: {val_results['auroc']:.4f} | "
              f"Time: {timedelta(seconds=int(epoch_time))}")
        
        # Early stopping check
        if val_results['loss'] < history['best_val_loss']:
            history['best_val_loss'] = val_results['loss']
            history['best_epoch'] = epoch + 1
            best_model_state = model.state_dict().copy()
            epochs_since_improvement = 0
            
            # Save checkpoint (unwrap DataParallel if needed)
            checkpoint_path = os.path.join(checkpoint_dir, f'seed{seed}_best.pt')
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_results['loss'],
                'val_auroc': val_results['auroc'],
            }, checkpoint_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    history['total_time'] = time.time() - total_start_time
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(
    dataset_name,
    dataset_dir,
    ood_dir,
    output_dir,
    pretrained_weights_path,
    config=None,
    seeds=None
):
    """
    Run full experiment for a single dataset with multiple seeds.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'juliet_c_simhash_k=1')
        dataset_dir (str): Path to the dataset directory
        ood_dir (str): Path to the OOD evaluation dataset
        output_dir (str): Path to save results
        pretrained_weights_path (str): Path to pretrained embedding weights
        config (dict, optional): Training configuration
        seeds (list, optional): List of random seeds
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    if seeds is None:
        seeds = SEEDS
    
    # Setup device for multi-GPU training
    if config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
        # Use all available GPUs
        device = torch.device('cuda:0')  # Primary device
        print(f"Multi-GPU training enabled")
        print(f"PyTorch sees {torch.cuda.device_count()} GPUs:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
    elif config.get('use_multi_gpu', False):
        # Multi-GPU requested but only 1 GPU available
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        print(f"WARNING: Multi-GPU requested but only {torch.cuda.device_count()} GPU(s) detected")
        print(f"Falling back to single GPU mode: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    else:
        # Single GPU mode
        device = select_best_gpu(min_free_gb=15)
        print(f"Using single GPU: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Create output directories
    experiment_dir = os.path.join(output_dir, dataset_name)
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    results_dir = os.path.join(experiment_dir, 'results')
    plots_dir = os.path.join(experiment_dir, 'plots')
    
    for d in [checkpoint_dir, results_dir, plots_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Load pretrained weights
    print(f"Loading pretrained weights from {pretrained_weights_path}...")
    pretrained = torch.load(pretrained_weights_path, map_location=device)
    pretrained_weights = pretrained['tok_embeddings.weight']
    print(f"Pretrained weights shape: {pretrained_weights.shape}")
    
    # Load data
    print(f"\nLoading dataset from {dataset_dir}...")
    train_loader = load_dataset(dataset_dir, 'train', config['batch_size'], shuffle=True)
    val_loader = load_dataset(dataset_dir, 'val', config['batch_size'], shuffle=False)
    test_loader = load_dataset(dataset_dir, 'test', config['batch_size'], shuffle=False)
    
    test_cwe_indices = load_cwe_indices(dataset_dir, 'test')
    idx_to_cwe = load_cwe_mapping(dataset_dir)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Load OOD dataset
    print(f"\nLoading OOD dataset from {ood_dir}...")
    ood_loader = load_dataset(ood_dir, 'test', config['batch_size'], shuffle=False)
    print(f"OOD test batches: {len(ood_loader)}")
    
    # Results storage
    all_results = {
        'dataset': dataset_name,
        'config': config,
        'seeds': {},
        'timestamp': datetime.now().isoformat(),
    }
    
    criterion = nn.BCELoss().to(device)
    
    # Train with each seed
    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"SEED {seed}")
        print(f"{'#'*60}")
        
        # Train model
        model, history = train_model(
            train_loader, val_loader, config, pretrained_weights,
            device, checkpoint_dir, seed
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
        print(f"Test Results: Loss={test_results['loss']:.4f}, AUROC={test_results['auroc']:.4f}, "
              f"F1={test_results['f1']:.4f}")
        
        # Per-CWE evaluation
        cwe_results = evaluate_per_cwe(test_preds, test_labels, test_cwe_indices, idx_to_cwe, device)
        
        # Print per-CWE summary
        if cwe_results:
            print(f"\nPer-CWE Results ({len(cwe_results)} CWEs evaluated):")
            # Sort by AUROC and show top/bottom 5
            sorted_cwes = sorted(cwe_results.items(), key=lambda x: x[1]['auroc'], reverse=True)
            print("  Top 5 CWEs:")
            for cwe, metrics in sorted_cwes[:5]:
                print(f"    {cwe}: AUROC={metrics['auroc']:.4f} (n={metrics['n_samples']}, pos={metrics['n_positive']})")
            if len(sorted_cwes) > 10:
                print("  ...")
            print("  Bottom 5 CWEs:")
            for cwe, metrics in sorted_cwes[-5:]:
                print(f"    {cwe}: AUROC={metrics['auroc']:.4f} (n={metrics['n_samples']}, pos={metrics['n_positive']})")
            
            # Compute CWE-level statistics
            cwe_aurocs = [m['auroc'] for m in cwe_results.values()]
            print(f"  CWE AUROC: Mean={np.mean(cwe_aurocs):.4f}, Std={np.std(cwe_aurocs):.4f}, "
                  f"Min={np.min(cwe_aurocs):.4f}, Max={np.max(cwe_aurocs):.4f}")
        
        # Evaluate on OOD set
        print("\nEvaluating on OOD set...")
        ood_results, _, _ = evaluate(model, ood_loader, criterion, device)
        print(f"OOD Results: Loss={ood_results['loss']:.4f}, AUROC={ood_results['auroc']:.4f}, "
              f"F1={ood_results['f1']:.4f}")
        
        # Store results
        all_results['seeds'][seed] = {
            'history': history,
            'test': test_results,
            'ood': ood_results,
            'per_cwe': cwe_results,
        }
        
        # Save model (unwrap DataParallel if needed)
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        model_path = os.path.join(results_dir, f'model_seed{seed}.pt')
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'config': model_to_save.get_config(),
            'test_results': test_results,
            'ood_results': ood_results,
            'history': history,
            'seed': seed,
        }, model_path)
        
        # Plot training curves for this seed
        plot_training_curves(history, seed, plots_dir)
        
        # Clean up GPU memory before next seed to prevent memory buildup
        del model
        clear_gpu_memory(verbose=True)
        print(f"Completed seed {seed}, GPU memory cleaned")
    
    # Compute aggregate statistics
    all_results['aggregate'] = compute_aggregate_stats(all_results['seeds'])
    
    # Save per-CWE results to a dedicated CSV file
    save_per_cwe_csv(all_results['seeds'], results_dir)
    
    # Save all results
    results_path = os.path.join(results_dir, 'experiment_results.json')
    with open(results_path, 'w') as f:
        # Convert non-serializable items
        json.dump(make_serializable(all_results), f, indent=2)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"\nAggregate Results:")
    print(f"  Test AUROC: {all_results['aggregate']['test_auroc_mean']:.4f} ± {all_results['aggregate']['test_auroc_std']:.4f}")
    print(f"  Test F1: {all_results['aggregate']['test_f1_mean']:.4f} ± {all_results['aggregate']['test_f1_std']:.4f}")
    print(f"  OOD AUROC: {all_results['aggregate']['ood_auroc_mean']:.4f} ± {all_results['aggregate']['ood_auroc_std']:.4f}")
    print(f"  OOD F1: {all_results['aggregate']['ood_f1_mean']:.4f} ± {all_results['aggregate']['ood_f1_std']:.4f}")
    
    return all_results


def compute_aggregate_stats(seeds_results):
    """Compute mean and std across seeds."""
    
    test_aurocs = [s['test']['auroc'] for s in seeds_results.values()]
    test_f1s = [s['test']['f1'] for s in seeds_results.values()]
    ood_aurocs = [s['ood']['auroc'] for s in seeds_results.values()]
    ood_f1s = [s['ood']['f1'] for s in seeds_results.values()]
    
    return {
        'test_auroc_mean': np.mean(test_aurocs),
        'test_auroc_std': np.std(test_aurocs),
        'test_f1_mean': np.mean(test_f1s),
        'test_f1_std': np.std(test_f1s),
        'ood_auroc_mean': np.mean(ood_aurocs),
        'ood_auroc_std': np.std(ood_aurocs),
        'ood_f1_mean': np.mean(ood_f1s),
        'ood_f1_std': np.std(ood_f1s),
    }


def save_per_cwe_csv(seeds_results, results_dir):
    """Save per-CWE results to a CSV file for analysis."""
    import csv
    
    csv_path = os.path.join(results_dir, 'per_cwe_results.csv')
    
    # Collect all CWEs across seeds
    all_cwes = set()
    for seed_data in seeds_results.values():
        if 'per_cwe' in seed_data:
            all_cwes.update(seed_data['per_cwe'].keys())
    
    if not all_cwes:
        print("  No per-CWE results to save")
        return
    
    # Build data for each CWE
    cwe_data = {}
    for cwe in sorted(all_cwes):
        aurocs = []
        n_samples = []
        n_positive = []
        for seed_data in seeds_results.values():
            if 'per_cwe' in seed_data and cwe in seed_data['per_cwe']:
                aurocs.append(seed_data['per_cwe'][cwe]['auroc'])
                n_samples.append(seed_data['per_cwe'][cwe]['n_samples'])
                n_positive.append(seed_data['per_cwe'][cwe]['n_positive'])
        
        if aurocs:
            cwe_data[cwe] = {
                'auroc_mean': np.mean(aurocs),
                'auroc_std': np.std(aurocs),
                'auroc_min': np.min(aurocs),
                'auroc_max': np.max(aurocs),
                'n_samples': int(np.mean(n_samples)),
                'n_positive': int(np.mean(n_positive)),
                'n_seeds': len(aurocs),
            }
    
    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['CWE', 'AUROC_Mean', 'AUROC_Std', 'AUROC_Min', 'AUROC_Max', 
                        'N_Samples', 'N_Positive', 'N_Seeds'])
        for cwe in sorted(cwe_data.keys()):
            d = cwe_data[cwe]
            writer.writerow([cwe, f"{d['auroc_mean']:.4f}", f"{d['auroc_std']:.4f}",
                           f"{d['auroc_min']:.4f}", f"{d['auroc_max']:.4f}",
                           d['n_samples'], d['n_positive'], d['n_seeds']])
    
    print(f"  Per-CWE results saved to: {csv_path}")
    print(f"  Total CWEs evaluated: {len(cwe_data)}")


def make_serializable(obj):
    """Convert non-JSON-serializable objects."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def plot_training_curves(history, seed, plots_dir):
    """Plot and save training curves."""
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
    axes[1].plot(epochs, history['val_auroc'], label='Validation AUROC')
    axes[1].axvline(x=history['best_epoch'], color='r', linestyle='--', alpha=0.5, label='Best')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUROC')
    axes[1].set_title(f'Validation AUROC (Seed {seed})')
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend()
    
    # Time
    axes[2].plot(epochs, history['train_times'], label='Train')
    axes[2].plot(epochs, history['eval_times'], label='Eval')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Time (s)')
    axes[2].set_title(f'Time per Epoch (Seed {seed})')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'training_seed{seed}.png'), dpi=150)
    plt.close()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train LSTM for vulnerability detection')
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to the training dataset directory')
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='Name of the dataset (e.g., juliet_c_simhash_k=1)')
    parser.add_argument('--ood-dir', type=str, required=True,
                        help='Path to the OOD evaluation dataset')
    parser.add_argument('--output-dir', type=str, default='experiments',
                        help='Output directory for results')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to pretrained embedding weights')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n-heads', type=int, default=8,
                        help='Number of attention heads (must divide hidden_dim*2 evenly)')
    parser.add_argument('--multi-gpu', action='store_true', default=True,
                        help='Use DataParallel for multi-GPU training (default: True)')
    parser.add_argument('--no-multi-gpu', dest='multi_gpu', action='store_false',
                        help='Disable multi-GPU training')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                        help='Random seeds to use')
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['patience'] = args.patience
    config['learning_rate'] = args.lr
    config['n_heads'] = args.n_heads
    config['use_multi_gpu'] = args.multi_gpu
    
    # Adjust batch size for multi-GPU training
    if config['use_multi_gpu'] and torch.cuda.device_count() > 1:
        effective_batch_size = config['batch_size'] * torch.cuda.device_count()
        print(f"Multi-GPU mode: effective batch size = {config['batch_size']} x {torch.cuda.device_count()} = {effective_batch_size}")
    
    run_experiment(
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
        ood_dir=args.ood_dir,
        output_dir=args.output_dir,
        pretrained_weights_path=args.weights,
        config=config,
        seeds=args.seeds
    )


if __name__ == '__main__':
    main()
