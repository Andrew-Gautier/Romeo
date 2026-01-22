"""
LSTM Training Script with DistributedDataParallel (DDP)
Trains models on Juliet C SimHash datasets (k=1 to k=12) with multiple seeds.
Evaluates on out-of-distribution (OOD) dataset (Devign).

This version uses DDP for efficient multi-GPU training.
Launch with: torchrun --nproc_per_node=8 train_lstm_ddp.py [args]
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os
import json
import time
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm

from classifier import LSTMClassifier, create_model


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'batch_size': 16,  # Per-GPU batch size (effective = batch_size * num_gpus)
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
    'n_heads': 8,
}

SEEDS = [42, 123, 456, 789, 1024]


# ============================================================================
# DDP Utilities
# ============================================================================

def setup_ddp():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size():
    """Get the number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    """Get the rank of the current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def print_rank0(*args, **kwargs):
    """Print only from rank 0."""
    if is_main_process():
        print(*args, **kwargs)


# ============================================================================
# Data Loading with Distributed Sampler
# ============================================================================

def load_dataset_ddp(data_dir, split='train', batch_size=32, num_workers=2):
    """
    Load a dataset split with DistributedSampler for DDP training.
    
    Args:
        data_dir (str): Path to dataset directory
        split (str): One of 'train', 'val', 'test'
        batch_size (int): Batch size per GPU
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (DataLoader, DistributedSampler or None)
    """
    sequences = torch.load(os.path.join(data_dir, f'{split}_sequences.pt')).long()
    labels = torch.load(os.path.join(data_dir, f'{split}_labels.pt'))
    
    dataset = TensorDataset(sequences, labels)
    
    # Use DistributedSampler for training data when using DDP
    if split == 'train' and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        shuffle = (split == 'train')
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        sampler=sampler,
        drop_last=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    
    return loader, sampler


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


def load_dataset_single_gpu(data_dir, split='test', batch_size=32, num_workers=2):
    """
    Load a dataset for single-GPU evaluation (no DistributedSampler).
    Used for final test/OOD evaluation on rank 0 to ensure correct sample ordering.
    
    Args:
        data_dir (str): Path to dataset directory
        split (str): One of 'train', 'val', 'test'
        batch_size (int): Batch size
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        DataLoader: Non-distributed dataloader
    """
    sequences = torch.load(os.path.join(data_dir, f'{split}_sequences.pt')).long()
    labels = torch.load(os.path.join(data_dir, f'{split}_labels.pt'))
    
    dataset = TensorDataset(sequences, labels)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Keep order for CWE alignment
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, iterator, optimizer, criterion, device, gradient_clip=1.0, sampler=None, epoch=0):
    """Train for one epoch with DDP."""
    epoch_loss = 0
    model.train()
    
    # Set epoch for sampler (important for proper shuffling across epochs)
    if sampler is not None:
        sampler.set_epoch(epoch)
    
    # Only show progress bar on rank 0
    iterator_wrapped = tqdm(iterator, desc='Training', leave=False) if is_main_process() else iterator
    
    for batch_sequences, batch_labels in iterator_wrapped:
        batch_sequences = batch_sequences.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True).float()
        
        optimizer.zero_grad()
        predictions = model(batch_sequences).squeeze(1)
        
        loss = criterion(predictions, batch_labels)
        loss.backward()
        
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    # Average loss across all processes
    avg_loss = epoch_loss / len(iterator)
    if dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
    
    return avg_loss


def evaluate(model, iterator, criterion, device):
    """Evaluate model on a dataset (run on all ranks, aggregate results)."""
    epoch_loss = 0
    model.eval()
    
    all_predictions = []
    all_labels = []
    num_batches = 0
    
    # Only show progress bar on rank 0
    iterator_wrapped = tqdm(iterator, desc='Evaluating', leave=False) if is_main_process() else iterator
    
    with torch.no_grad():
        for batch_sequences, batch_labels in iterator_wrapped:
            batch_sequences = batch_sequences.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True).float()
            
            predictions = model(batch_sequences).squeeze(1)
            
            loss = criterion(predictions, batch_labels)
            epoch_loss += loss.item()
            num_batches += 1
            
            all_predictions.append(predictions.detach())
            all_labels.append(batch_labels.detach())
    
    # Concatenate local predictions (on GPU)
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    # Gather predictions from all ranks using CUDA tensors
    if dist.is_initialized():
        world_size = dist.get_world_size()
        
        # Get sizes from all ranks
        local_size = torch.tensor([all_predictions.size(0)], device=device)
        all_sizes = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        all_sizes = [int(s.item()) for s in all_sizes]
        max_size = max(all_sizes)
        
        # Pad tensors to max size
        padded_predictions = torch.zeros(max_size, device=device)
        padded_labels = torch.zeros(max_size, device=device)
        padded_predictions[:all_predictions.size(0)] = all_predictions
        padded_labels[:all_labels.size(0)] = all_labels
        
        # Gather from all ranks
        gathered_predictions = [torch.zeros(max_size, device=device) for _ in range(world_size)]
        gathered_labels = [torch.zeros(max_size, device=device) for _ in range(world_size)]
        dist.all_gather(gathered_predictions, padded_predictions)
        dist.all_gather(gathered_labels, padded_labels)
        
        # Concatenate and trim to actual sizes
        all_predictions_list = []
        all_labels_list = []
        for i in range(world_size):
            all_predictions_list.append(gathered_predictions[i][:all_sizes[i]])
            all_labels_list.append(gathered_labels[i][:all_sizes[i]])
        
        all_predictions = torch.cat(all_predictions_list)
        all_labels = torch.cat(all_labels_list)
        
        # Average loss across ranks
        loss_tensor = torch.tensor([epoch_loss / num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
    else:
        avg_loss = epoch_loss / num_batches
    
    # Move to CPU for metric computation
    all_predictions_cpu = all_predictions.cpu().numpy()
    all_labels_cpu = all_labels.cpu().numpy()
    binary_preds = (all_predictions_cpu >= 0.5).astype(float)
    
    # Compute metrics using sklearn (no DDP sync issues)
    try:
        auroc = roc_auc_score(all_labels_cpu, all_predictions_cpu)
    except ValueError:
        auroc = 0.5  # Handle case with single class
    
    accuracy = accuracy_score(all_labels_cpu, binary_preds)
    precision = precision_score(all_labels_cpu, binary_preds, zero_division=0)
    recall = recall_score(all_labels_cpu, binary_preds, zero_division=0)
    f1 = f1_score(all_labels_cpu, binary_preds, zero_division=0)
    
    results = {
        'loss': avg_loss,
        'auroc': auroc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
    return results, torch.from_numpy(all_predictions_cpu), torch.from_numpy(all_labels_cpu)


def evaluate_single_gpu(model, iterator, criterion, device):
    """
    Evaluate model on a single GPU without DDP gathering.
    Used for final test/OOD evaluation to maintain sample ordering for CWE analysis.
    
    Args:
        model: The model (can be DDP-wrapped, will extract underlying module)
        iterator: DataLoader for evaluation
        criterion: Loss function
        device: Device to run on
        
    Returns:
        tuple: (results_dict, predictions_tensor, labels_tensor)
    """
    epoch_loss = 0
    
    # Get the underlying model if DDP-wrapped
    eval_model = model.module if hasattr(model, 'module') else model
    eval_model.eval()
    
    all_predictions = []
    all_labels = []
    
    iterator_wrapped = tqdm(iterator, desc='Evaluating', leave=False)
    
    with torch.no_grad():
        for batch_sequences, batch_labels in iterator_wrapped:
            batch_sequences = batch_sequences.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True).float()
            
            predictions = eval_model(batch_sequences).squeeze(1)
            
            loss = criterion(predictions, batch_labels)
            epoch_loss += loss.item()
            
            all_predictions.append(predictions.cpu())
            all_labels.append(batch_labels.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    # Convert to numpy for sklearn
    all_predictions_np = all_predictions.numpy()
    all_labels_np = all_labels.numpy()
    binary_preds = (all_predictions_np >= 0.5).astype(float)
    
    # Compute metrics
    try:
        auroc = roc_auc_score(all_labels_np, all_predictions_np)
    except ValueError:
        auroc = 0.5
    
    accuracy = accuracy_score(all_labels_np, binary_preds)
    precision = precision_score(all_labels_np, binary_preds, zero_division=0)
    recall = recall_score(all_labels_np, binary_preds, zero_division=0)
    f1 = f1_score(all_labels_np, binary_preds, zero_division=0)
    
    results = {
        'loss': epoch_loss / len(iterator),
        'auroc': auroc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
    return results, all_predictions, all_labels


def evaluate_per_cwe(predictions, labels, cwe_indices, idx_to_cwe, device):
    """Evaluate metrics per CWE category."""
    if cwe_indices is None or idx_to_cwe is None:
        return {}
    
    cwe_results = {}
    unique_cwes = torch.unique(cwe_indices)
    
    # Convert to numpy for sklearn
    predictions_np = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    cwe_indices_np = cwe_indices.numpy() if isinstance(cwe_indices, torch.Tensor) else cwe_indices
    
    for cwe_idx in unique_cwes:
        cwe_idx = cwe_idx.item()
        mask = cwe_indices_np == cwe_idx
        
        if mask.sum() < 2:
            continue
        
        cwe_preds = predictions_np[mask]
        cwe_labels = labels_np[mask]
        
        if len(np.unique(cwe_labels)) < 2:
            continue
        
        try:
            auroc = roc_auc_score(cwe_labels, cwe_preds)
        except ValueError:
            auroc = 0.5  # Handle edge cases
        
        cwe_name = idx_to_cwe.get(cwe_idx, f'CWE_{cwe_idx}')
        cwe_results[cwe_name] = {
            'auroc': auroc,
            'n_samples': int(mask.sum()),
            'n_positive': int(cwe_labels.sum()),
        }
    
    return cwe_results
    
    return cwe_results


# ============================================================================
# Main Training Loop
# ============================================================================

def train_model(
    train_loader,
    train_sampler,
    val_loader, 
    config, 
    pretrained_weights,
    device,
    local_rank,
    checkpoint_dir,
    seed
):
    """
    Train a model with early stopping using DDP.
    
    Returns:
        tuple: (model, training_history)
    """
    # Set seed for reproducibility (different per rank for data augmentation diversity)
    torch.manual_seed(seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + get_rank())
    
    # Create model on local device
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
    
    # Wrap model in DDP
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print_rank0(f"Using DistributedDataParallel with {get_world_size()} GPUs")
    
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
    
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Training with seed {seed}")
    print_rank0(f"{'='*60}")
    
    total_start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Train
        train_start = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            device, config['gradient_clip'], train_sampler, epoch
        )
        train_time = time.time() - train_start
        
        # Validate (on rank 0 only for metrics)
        eval_start = time.time()
        val_results, _, _ = evaluate(model, val_loader, criterion, device)
        eval_time = time.time() - eval_start
        
        # Broadcast validation results to all ranks for consistent early stopping
        if dist.is_initialized():
            val_loss_tensor = torch.tensor([val_results.get('loss', 0)], device=device)
            dist.broadcast(val_loss_tensor, src=0)
            val_results['loss'] = val_loss_tensor.item()
            
            if is_main_process():
                val_auroc_tensor = torch.tensor([val_results.get('auroc', 0)], device=device)
            else:
                val_auroc_tensor = torch.tensor([0.0], device=device)
            dist.broadcast(val_auroc_tensor, src=0)
            val_results['auroc'] = val_auroc_tensor.item()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_results['loss'])
        history['val_auroc'].append(val_results.get('auroc', 0))
        history['train_times'].append(train_time)
        history['eval_times'].append(eval_time)
        
        epoch_time = time.time() - epoch_start
        
        print_rank0(f"Epoch {epoch+1}/{config['epochs']} | "
                   f"Train Loss: {train_loss:.4f} | "
                   f"Val Loss: {val_results['loss']:.4f} | "
                   f"Val AUROC: {val_results.get('auroc', 0):.4f} | "
                   f"Time: {timedelta(seconds=int(epoch_time))}")
        
        # Early stopping check (consistent across all ranks)
        if val_results['loss'] < history['best_val_loss']:
            history['best_val_loss'] = val_results['loss']
            history['best_epoch'] = epoch + 1
            epochs_since_improvement = 0
            
            # Save checkpoint from rank 0 only
            if is_main_process():
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
                checkpoint_path = os.path.join(checkpoint_dir, f'seed{seed}_best.pt')
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_results['loss'],
                    'val_auroc': val_results.get('auroc', 0),
                }, checkpoint_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= config['patience']:
                print_rank0(f"Early stopping at epoch {epoch+1}")
                break
        
        # Synchronize all processes at end of epoch
        if dist.is_initialized():
            dist.barrier()
    
    history['total_time'] = time.time() - total_start_time
    
    # Load best model state
    if best_model_state is not None and is_main_process():
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
    local_rank,
    config=None,
    seeds=None
):
    """
    Run full experiment for a single dataset with multiple seeds using DDP.
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    if seeds is None:
        seeds = SEEDS
    
    device = torch.device(f'cuda:{local_rank}')
    
    # Print info from rank 0 only
    print_rank0(f"\n{'='*60}")
    print_rank0(f"DDP Training: {dataset_name}")
    print_rank0(f"{'='*60}")
    print_rank0(f"World size: {get_world_size()} GPUs")
    print_rank0(f"Batch size per GPU: {config['batch_size']}")
    print_rank0(f"Effective batch size: {config['batch_size'] * get_world_size()}")
    
    if is_main_process():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create output directories (rank 0 only)
    experiment_dir = os.path.join(output_dir, dataset_name)
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    results_dir = os.path.join(experiment_dir, 'results')
    plots_dir = os.path.join(experiment_dir, 'plots')
    
    if is_main_process():
        for d in [checkpoint_dir, results_dir, plots_dir]:
            os.makedirs(d, exist_ok=True)
    
    # Wait for directories to be created
    if dist.is_initialized():
        dist.barrier()
    
    # Load pretrained weights
    print_rank0(f"Loading pretrained weights from {pretrained_weights_path}...")
    pretrained = torch.load(pretrained_weights_path, map_location='cpu')
    pretrained_weights = pretrained['tok_embeddings.weight']
    print_rank0(f"Pretrained weights shape: {pretrained_weights.shape}")
    
    # Load data with DistributedSampler
    print_rank0(f"\nLoading dataset from {dataset_dir}...")
    train_loader, train_sampler = load_dataset_ddp(dataset_dir, 'train', config['batch_size'])
    val_loader, _ = load_dataset_ddp(dataset_dir, 'val', config['batch_size'])
    
    # For test/OOD evaluation, load without DistributedSampler on rank 0 only
    # This ensures correct sample ordering for CWE analysis
    if is_main_process():
        test_loader_single = load_dataset_single_gpu(dataset_dir, 'test', config['batch_size'] * 2)
        ood_loader_single = load_dataset_single_gpu(ood_dir, 'test', config['batch_size'] * 2)
    else:
        test_loader_single = None
        ood_loader_single = None
    
    test_cwe_indices = load_cwe_indices(dataset_dir, 'test')
    idx_to_cwe = load_cwe_mapping(dataset_dir)
    
    print_rank0(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Load OOD dataset info
    print_rank0(f"\nLoading OOD dataset from {ood_dir}...")
    if is_main_process():
        print(f"Test batches (single GPU): {len(test_loader_single)}, OOD batches: {len(ood_loader_single)}")
    
    # Results storage
    all_results = {
        'dataset': dataset_name,
        'config': config,
        'world_size': get_world_size(),
        'effective_batch_size': config['batch_size'] * get_world_size(),
        'seeds': {},
        'timestamp': datetime.now().isoformat(),
    }
    
    criterion = nn.BCELoss().to(device)
    
    # Train with each seed
    for seed in seeds:
        print_rank0(f"\n{'#'*60}")
        print_rank0(f"SEED {seed}")
        print_rank0(f"{'#'*60}")
        
        # Train model
        model, history = train_model(
            train_loader, train_sampler, val_loader, config, pretrained_weights,
            device, local_rank, checkpoint_dir, seed
        )
        
        # Synchronize after training before evaluation
        if dist.is_initialized():
            dist.barrier()
        
        # Final evaluation on rank 0 only (single GPU, correct sample ordering)
        if is_main_process():
            print("\nEvaluating on test set (single GPU)...")
            test_results, test_preds, test_labels = evaluate_single_gpu(
                model, test_loader_single, criterion, device
            )
            print(f"Test Results: Loss={test_results['loss']:.4f}, AUROC={test_results['auroc']:.4f}, "
                  f"F1={test_results['f1']:.4f}")
            
            # Per-CWE evaluation
            cwe_results = evaluate_per_cwe(test_preds, test_labels, test_cwe_indices, idx_to_cwe, device)
            
            if cwe_results:
                print(f"\nPer-CWE Results ({len(cwe_results)} CWEs evaluated):")
                sorted_cwes = sorted(cwe_results.items(), key=lambda x: x[1]['auroc'], reverse=True)
                print("  Top 5 CWEs:")
                for cwe, metrics in sorted_cwes[:5]:
                    print(f"    {cwe}: AUROC={metrics['auroc']:.4f} (n={metrics['n_samples']})")
                if len(sorted_cwes) > 10:
                    print("  ...")
                print("  Bottom 5 CWEs:")
                for cwe, metrics in sorted_cwes[-5:]:
                    print(f"    {cwe}: AUROC={metrics['auroc']:.4f} (n={metrics['n_samples']})")
                
                cwe_aurocs = [m['auroc'] for m in cwe_results.values()]
                print(f"  CWE AUROC: Mean={np.mean(cwe_aurocs):.4f}, Std={np.std(cwe_aurocs):.4f}")
            
            # Evaluate on OOD set
            print("\nEvaluating on OOD set (single GPU)...")
            ood_results, _, _ = evaluate_single_gpu(model, ood_loader_single, criterion, device)
            print(f"OOD Results: Loss={ood_results['loss']:.4f}, AUROC={ood_results['auroc']:.4f}, "
                  f"F1={ood_results['f1']:.4f}")
            
            # Store results
            all_results['seeds'][seed] = {
                'history': history,
                'test': test_results,
                'ood': ood_results,
                'per_cwe': cwe_results,
            }
            
            # Save model
            model_to_save = model.module if hasattr(model, 'module') else model
            model_path = os.path.join(results_dir, f'model_seed{seed}.pt')
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'config': model_to_save.get_config(),
                'test_results': test_results,
                'ood_results': ood_results,
                'history': history,
                'seed': seed,
            }, model_path)
            
            # Plot training curves
            plot_training_curves(history, seed, plots_dir)
        
        # Clean up model before next seed
        del model
        torch.cuda.empty_cache()
        
        # Synchronize before next seed
        if dist.is_initialized():
            dist.barrier()
        
        print_rank0(f"Completed seed {seed}")
    
    # Final aggregation and saving (rank 0 only)
    if is_main_process():
        all_results['aggregate'] = compute_aggregate_stats(all_results['seeds'])
        save_per_cwe_csv(all_results['seeds'], results_dir)
        
        results_path = os.path.join(results_dir, 'experiment_results.json')
        with open(results_path, 'w') as f:
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
    """Save per-CWE results to a CSV file."""
    import csv
    
    csv_path = os.path.join(results_dir, 'per_cwe_results.csv')
    
    all_cwes = set()
    for seed_data in seeds_results.values():
        if 'per_cwe' in seed_data:
            all_cwes.update(seed_data['per_cwe'].keys())
    
    if not all_cwes:
        print("  No per-CWE results to save")
        return
    
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
    
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'], label='Validation')
    axes[0].axvline(x=history['best_epoch'], color='r', linestyle='--', alpha=0.5, label='Best')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Loss (Seed {seed})')
    axes[0].legend()
    
    axes[1].plot(epochs, history['val_auroc'], label='Validation AUROC')
    axes[1].axvline(x=history['best_epoch'], color='r', linestyle='--', alpha=0.5, label='Best')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUROC')
    axes[1].set_title(f'Validation AUROC (Seed {seed})')
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend()
    
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
    parser = argparse.ArgumentParser(description='Train LSTM with DDP for vulnerability detection')
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
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size per GPU (default: 16)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    
    args = parser.parse_args()
    
    # Setup DDP
    local_rank = setup_ddp()
    
    try:
        config = DEFAULT_CONFIG.copy()
        config['batch_size'] = args.batch_size
        config['epochs'] = args.epochs
        config['patience'] = args.patience
        config['learning_rate'] = args.lr
        config['n_heads'] = args.n_heads
        
        run_experiment(
            dataset_name=args.dataset_name,
            dataset_dir=args.dataset_dir,
            ood_dir=args.ood_dir,
            output_dir=args.output_dir,
            pretrained_weights_path=args.weights,
            local_rank=local_rank,
            config=config,
            seeds=args.seeds
        )
    finally:
        cleanup_ddp()


if __name__ == '__main__':
    main()
