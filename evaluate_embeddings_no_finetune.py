"""
Embedding Evaluation Script (No Fine-Tuning)

Evaluates the LSTM classifier with pretrained embeddings from 3 different models
on the Devign dataset WITHOUT any fine-tuning. This measures what the raw 
embedding representations give us for binary vulnerability classification
when only the randomly-initialized LSTM + attention head are used.

Models evaluated:
  1. aiXcoder-7B  (loaded from .pt file)
  2. DeepSeek-Coder-6.7B (loaded from HuggingFace cache)
  3. CodeLlama-7B (loaded from HuggingFace cache)

For each model:
  - The embedding layer is initialized with pretrained weights (frozen)
  - The LSTM, attention, and classification head are randomly initialized
  - The model is evaluated on Devign test split across multiple seeds
  - No training is performed

Usage:
  Single GPU:
    python evaluate_embeddings_no_finetune.py \
        --devign-dir /path/to/devign_tensors \
        --aixcoder-weights /path/to/aix3-7b-base.pt \
        --hf-cache-dir /path/to/huggingface \
        --output-dir /path/to/output \
        --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os
import glob
import json
import csv
import time
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm

# HuggingFace imports for loading model weights
from safetensors.torch import load_file as load_safetensors

from binary_classifier import LSTMClassifier, create_model


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'batch_size': 32,
    'lstm_nodes': 256,
    'vocab_size': 49152,       # Will be updated per model
    'embedding_size': 4096,    # Will be updated per model
    'output_dim': 1,
    'n_layers': 2,
    'bidirectional': True,
    'dropout': 0.5,
    'n_heads': 8,
}

# HuggingFace model configs (same as training scripts)
HF_MODEL_CONFIGS = {
    'deepseek-coder': {
        'embedding_key': 'model.embed_tokens.weight',
        'model_id': 'deepseek-ai/deepseek-coder-6.7b-base',
    },
    'codellama': {
        'embedding_key': 'model.embed_tokens.weight',
        'model_id': 'codellama/CodeLlama-7b-hf',
    },
}

SEEDS = list(range(1, 21))  # 20 seeds by default


# ============================================================================
# Embedding Loading Utilities
# ============================================================================

def load_aixcoder_embeddings(weights_path):
    """
    Load pretrained embeddings from aiXcoder .pt file.
    
    Args:
        weights_path: Path to aiXcoder weights file (e.g., aix3-7b-base.pt)
    
    Returns:
        tuple: (embedding_weights, vocab_size, embedding_dim)
    """
    print(f"Loading aiXcoder embeddings from: {weights_path}")
    pretrained = torch.load(weights_path, map_location='cpu')
    embeddings = pretrained['tok_embeddings.weight']
    vocab_size, embedding_dim = embeddings.shape
    print(f"  Shape: {embeddings.shape} (vocab={vocab_size}, dim={embedding_dim})")
    return embeddings, vocab_size, embedding_dim


def find_hf_model_path(cache_dir, model_name):
    """Find the model snapshot directory in the HuggingFace cache."""
    if model_name not in HF_MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(HF_MODEL_CONFIGS.keys())}")
    
    model_id = HF_MODEL_CONFIGS[model_name]['model_id']
    cache_name = f"models--{model_id.replace('/', '--')}"
    model_cache_dir = os.path.join(cache_dir, cache_name)
    
    if not os.path.exists(model_cache_dir):
        raise FileNotFoundError(f"Model cache not found: {model_cache_dir}")
    
    snapshots_dir = os.path.join(model_cache_dir, 'snapshots')
    if not os.path.exists(snapshots_dir):
        raise FileNotFoundError(f"Snapshots directory not found: {snapshots_dir}")
    
    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in: {snapshots_dir}")
    
    return os.path.join(snapshots_dir, snapshots[0])


def load_hf_embeddings(cache_dir, model_name):
    """
    Load pretrained embeddings from a HuggingFace model.
    
    Args:
        cache_dir: Path to HuggingFace cache directory
        model_name: Model identifier ('deepseek-coder' or 'codellama')
    
    Returns:
        tuple: (embedding_weights, vocab_size, embedding_dim)
    """
    model_path = find_hf_model_path(cache_dir, model_name)
    print(f"Loading {model_name} embeddings from: {model_path}")
    
    # Load config to get vocab size and embedding dim
    config_path = os.path.join(model_path, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    vocab_size = config.get('vocab_size', 32256)
    embedding_dim = config.get('hidden_size', 4096)
    print(f"  Config: vocab_size={vocab_size}, hidden_size={embedding_dim}")
    
    # Find and load the safetensors file(s)
    safetensor_files = glob.glob(os.path.join(model_path, '*.safetensors'))
    
    if not safetensor_files:
        # Fallback to .bin files
        bin_files = glob.glob(os.path.join(model_path, 'pytorch_model*.bin'))
        if bin_files:
            embedding_key = HF_MODEL_CONFIGS[model_name]['embedding_key']
            for bin_file in bin_files:
                state_dict = torch.load(bin_file, map_location='cpu')
                if embedding_key in state_dict:
                    embeddings = state_dict[embedding_key]
                    print(f"  Shape: {embeddings.shape}")
                    return embeddings, vocab_size, embedding_dim
        raise FileNotFoundError(f"No safetensors or .bin files found in: {model_path}")
    
    # Load embeddings from safetensors
    embedding_key = HF_MODEL_CONFIGS[model_name]['embedding_key']
    embeddings = None
    
    for sf_file in safetensor_files:
        try:
            state_dict = load_safetensors(sf_file)
            if embedding_key in state_dict:
                embeddings = state_dict[embedding_key]
                print(f"  Found in {os.path.basename(sf_file)}, shape: {embeddings.shape}")
                break
        except Exception as e:
            continue
    
    if embeddings is None:
        # Try alternative keys
        for sf_file in safetensor_files:
            state_dict = load_safetensors(sf_file)
            for key in state_dict.keys():
                if 'embed' in key.lower() and 'token' in key.lower():
                    embeddings = state_dict[key]
                    print(f"  Found via alt key '{key}', shape: {embeddings.shape}")
                    break
            if embeddings is not None:
                break
    
    if embeddings is None:
        raise KeyError(f"Could not find embedding weights for {model_name}")
    
    return embeddings, vocab_size, embedding_dim


# ============================================================================
# Data Loading
# ============================================================================

def load_dataset(data_dir, split='test', batch_size=32):
    """
    Load a dataset split for evaluation.
    
    Args:
        data_dir: Path to dataset directory containing {split}_sequences.pt and {split}_labels.pt
        split: One of 'train', 'val', 'test'
        batch_size: Batch size for DataLoader
    
    Returns:
        DataLoader
    """
    seq_path = os.path.join(data_dir, f'{split}_sequences.pt')
    label_path = os.path.join(data_dir, f'{split}_labels.pt')
    
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"Sequences not found: {seq_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Labels not found: {label_path}")
    
    sequences = torch.load(seq_path, map_location='cpu').long()
    labels = torch.load(label_path, map_location='cpu')
    
    print(f"  {split}: {sequences.shape[0]} samples, seq_len={sequences.shape[1]}")
    
    dataset = torch.utils.data.TensorDataset(sequences, labels)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    
    return loader


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate model on a dataset (single GPU, no DDP).
    
    Args:
        model: The LSTMClassifier model
        data_loader: DataLoader for evaluation
        criterion: Loss function
        device: Device to run on
    
    Returns:
        dict: Metrics (loss, auroc, accuracy, precision, recall, f1)
    """
    model.eval()
    epoch_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_sequences, batch_labels in tqdm(data_loader, desc='Evaluating', leave=False):
            batch_sequences = batch_sequences.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True).float()
            
            predictions = model(batch_sequences).squeeze(1)
            loss = criterion(predictions, batch_labels)
            epoch_loss += loss.item()
            
            all_predictions.append(predictions.cpu())
            all_labels.append(batch_labels.cpu())
    
    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()
    binary_preds = (all_predictions >= 0.5).astype(float)
    
    # Compute metrics
    try:
        auroc = roc_auc_score(all_labels, all_predictions)
    except ValueError:
        auroc = 0.5
    
    accuracy = accuracy_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds, zero_division=0)
    recall = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    
    results = {
        'loss': epoch_loss / len(data_loader),
        'auroc': auroc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
    return results


def run_no_finetune_evaluation(
    model_label,
    embedding_weights,
    vocab_size,
    embedding_dim,
    test_loader,
    device,
    config,
    seeds,
    freeze_embeddings=True,
):
    """
    Evaluate a model with pretrained embeddings and NO fine-tuning.
    
    For each seed:
      1. Create a fresh model with random LSTM/attention/FC weights
      2. Load pretrained embeddings into the embedding layer
      3. Optionally freeze the embedding layer
      4. Evaluate immediately on the test set (no training)
    
    Args:
        model_label: Human-readable name for this model
        embedding_weights: Pretrained embedding tensor
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension
        test_loader: DataLoader for the test set
        device: Device to run on
        config: Model config dict
        seeds: List of random seeds
        freeze_embeddings: If True, freeze embedding layer gradients
    
    Returns:
        dict: Results for all seeds + aggregate statistics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_label}")
    print(f"  Vocab size: {vocab_size}, Embedding dim: {embedding_dim}")
    print(f"  Freeze embeddings: {freeze_embeddings}")
    print(f"  Seeds: {len(seeds)}")
    print(f"{'='*60}")
    
    criterion = nn.BCELoss().to(device)
    
    all_seed_results = {}
    
    for seed in seeds:
        print(f"\n  Seed {seed}...")
        
        # Set seed for reproducible random initialization of LSTM/attention/FC
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Create fresh model with pretrained embeddings
        model_config = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': config['lstm_nodes'],
            'output_dim': config['output_dim'],
            'n_layers': config['n_layers'],
            'bidirectional': config['bidirectional'],
            'dropout': config['dropout'],
            'n_heads': config['n_heads'],
        }
        
        model = create_model(
            model_config,
            pretrained_weights=embedding_weights,
            device=device,
        )
        
        # Freeze embedding layer so it's purely the pretrained representation
        if freeze_embeddings:
            model.embedding.weight.requires_grad = False
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        if seed == seeds[0]:
            print(f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable, {frozen_params:,} frozen")
        
        # Evaluate immediately (no training!)
        results = evaluate_model(model, test_loader, criterion, device)
        
        all_seed_results[seed] = results
        print(f"    AUROC={results['auroc']:.4f}, F1={results['f1']:.4f}, "
              f"Acc={results['accuracy']:.4f}, Prec={results['precision']:.4f}, "
              f"Rec={results['recall']:.4f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Compute aggregate statistics
    metrics = ['auroc', 'f1', 'accuracy', 'precision', 'recall', 'loss']
    aggregate = {}
    for metric in metrics:
        values = [r[metric] for r in all_seed_results.values()]
        aggregate[f'{metric}_mean'] = float(np.mean(values))
        aggregate[f'{metric}_std'] = float(np.std(values))
        aggregate[f'{metric}_min'] = float(np.min(values))
        aggregate[f'{metric}_max'] = float(np.max(values))
    
    print(f"\n  Aggregate ({len(seeds)} seeds):")
    print(f"    AUROC: {aggregate['auroc_mean']:.4f} ± {aggregate['auroc_std']:.4f}")
    print(f"    F1:    {aggregate['f1_mean']:.4f} ± {aggregate['f1_std']:.4f}")
    print(f"    Acc:   {aggregate['accuracy_mean']:.4f} ± {aggregate['accuracy_std']:.4f}")
    print(f"    Prec:  {aggregate['precision_mean']:.4f} ± {aggregate['precision_std']:.4f}")
    print(f"    Rec:   {aggregate['recall_mean']:.4f} ± {aggregate['recall_std']:.4f}")
    
    return {
        'model': model_label,
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'freeze_embeddings': freeze_embeddings,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'seeds': {int(k): v for k, v in all_seed_results.items()},
        'aggregate': aggregate,
    }


# ============================================================================
# Output / Reporting
# ============================================================================

def save_results(all_model_results, output_dir, dataset_info):
    """Save all results to JSON, CSV, and generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # ---- JSON ----
    json_path = os.path.join(output_dir, 'no_finetune_results.json')
    output = {
        'experiment': 'Embedding Evaluation (No Fine-Tuning)',
        'description': 'Evaluates pretrained embeddings with randomly initialized LSTM/attention/FC head',
        'dataset': dataset_info,
        'timestamp': datetime.now().isoformat(),
        'models': {r['model']: r for r in all_model_results},
    }
    
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON results saved to: {json_path}")
    
    # ---- Per-seed CSV ----
    csv_path = os.path.join(output_dir, 'no_finetune_per_seed.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'seed', 'auroc', 'f1', 'accuracy', 'precision', 'recall', 'loss'])
        for result in all_model_results:
            for seed, metrics in sorted(result['seeds'].items()):
                writer.writerow([
                    result['model'], seed,
                    f"{metrics['auroc']:.6f}",
                    f"{metrics['f1']:.6f}",
                    f"{metrics['accuracy']:.6f}",
                    f"{metrics['precision']:.6f}",
                    f"{metrics['recall']:.6f}",
                    f"{metrics['loss']:.6f}",
                ])
    print(f"Per-seed CSV saved to: {csv_path}")
    
    # ---- Summary CSV ----
    summary_csv_path = os.path.join(output_dir, 'no_finetune_summary.csv')
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model', 'vocab_size', 'embedding_dim', 'total_params', 'trainable_params',
            'auroc_mean', 'auroc_std', 'f1_mean', 'f1_std',
            'accuracy_mean', 'accuracy_std', 'precision_mean', 'precision_std',
            'recall_mean', 'recall_std', 'loss_mean', 'loss_std',
        ])
        for r in all_model_results:
            agg = r['aggregate']
            writer.writerow([
                r['model'], r['vocab_size'], r['embedding_dim'],
                r['total_params'], r['trainable_params'],
                f"{agg['auroc_mean']:.6f}", f"{agg['auroc_std']:.6f}",
                f"{agg['f1_mean']:.6f}", f"{agg['f1_std']:.6f}",
                f"{agg['accuracy_mean']:.6f}", f"{agg['accuracy_std']:.6f}",
                f"{agg['precision_mean']:.6f}", f"{agg['precision_std']:.6f}",
                f"{agg['recall_mean']:.6f}", f"{agg['recall_std']:.6f}",
                f"{agg['loss_mean']:.6f}", f"{agg['loss_std']:.6f}",
            ])
    print(f"Summary CSV saved to: {summary_csv_path}")
    
    # ---- Summary text ----
    summary_path = os.path.join(output_dir, 'no_finetune_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EMBEDDING EVALUATION (NO FINE-TUNING) - SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_info}\n")
        f.write(f"Seeds: {len(all_model_results[0]['seeds'])}\n")
        f.write(f"Embeddings frozen: Yes\n\n")
        
        for r in all_model_results:
            agg = r['aggregate']
            f.write(f"--- {r['model']} ---\n")
            f.write(f"  Vocab: {r['vocab_size']:,}, Dim: {r['embedding_dim']}\n")
            f.write(f"  Params: {r['total_params']:,} total, {r['trainable_params']:,} trainable\n")
            f.write(f"  AUROC:     {agg['auroc_mean']:.4f} ± {agg['auroc_std']:.4f}  (range: {agg['auroc_min']:.4f} - {agg['auroc_max']:.4f})\n")
            f.write(f"  F1:        {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}  (range: {agg['f1_min']:.4f} - {agg['f1_max']:.4f})\n")
            f.write(f"  Accuracy:  {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}\n")
            f.write(f"  Precision: {agg['precision_mean']:.4f} ± {agg['precision_std']:.4f}\n")
            f.write(f"  Recall:    {agg['recall_mean']:.4f} ± {agg['recall_std']:.4f}\n")
            f.write(f"  Loss:      {agg['loss_mean']:.4f} ± {agg['loss_std']:.4f}\n\n")
    print(f"Summary text saved to: {summary_path}")
    
    # ---- Comparison bar chart ----
    generate_comparison_plot(all_model_results, output_dir)
    
    return output


def generate_comparison_plot(all_model_results, output_dir):
    """Generate publication-quality comparison bar charts."""
    models = [r['model'] for r in all_model_results]
    metrics = ['auroc', 'f1', 'accuracy', 'precision', 'recall']
    metric_labels = ['AUROC', 'F1 Score', 'Accuracy', 'Precision', 'Recall']
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        means = [r['aggregate'][f'{metric}_mean'] for r in all_model_results]
        stds = [r['aggregate'][f'{metric}_std'] for r in all_model_results]
        
        x = np.arange(len(models))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(models)],
                      edgecolor='black', linewidth=0.5, alpha=0.85)
        
        ax.set_ylabel(label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=9)
        ax.set_title(label, fontsize=13, fontweight='bold')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std + 0.005,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Set y-axis limits
        if metric in ['auroc', 'accuracy']:
            ax.set_ylim(0.0, 1.05)
        else:
            ax.set_ylim(0.0, max(means) * 1.3 + 0.05)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    plt.suptitle('Embedding Evaluation — No Fine-Tuning (Devign Dataset)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    for ext in ['png', 'pdf']:
        plot_path = os.path.join(output_dir, f'no_finetune_comparison.{ext}')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {output_dir}/no_finetune_comparison.png/pdf")
    
    # ---- Per-seed scatter/box plot ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (metric, label) in enumerate([('auroc', 'AUROC'), ('f1', 'F1 Score')]):
        ax = axes2[idx]
        data_for_box = []
        for r in all_model_results:
            values = [r['seeds'][s][metric] for s in sorted(r['seeds'].keys())]
            data_for_box.append(values)
        
        bp = ax.boxplot(data_for_box, labels=[m.replace('-', '\n') for m in models],
                       patch_artist=True, widths=0.5)
        
        for patch, color in zip(bp['boxes'], colors[:len(models)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Overlay individual seed points
        for i, (r, color) in enumerate(zip(all_model_results, colors)):
            values = [r['seeds'][s][metric] for s in sorted(r['seeds'].keys())]
            jitter = np.random.normal(0, 0.04, len(values))
            ax.scatter([i + 1 + j for j in jitter], values, color=color,
                      alpha=0.7, s=20, zorder=3, edgecolors='black', linewidths=0.3)
        
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} Distribution Across Seeds', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    plt.suptitle('Per-Seed Variability — No Fine-Tuning',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    for ext in ['png', 'pdf']:
        plot_path = os.path.join(output_dir, f'no_finetune_boxplot.{ext}')
        fig2.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Box plot saved to: {output_dir}/no_finetune_boxplot.png/pdf")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate pretrained embeddings with no fine-tuning on Devign',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--devign-dir', type=str, required=True,
                        help='Path to Devign tensor directory (containing test_sequences.pt, test_labels.pt)')
    parser.add_argument('--aixcoder-weights', type=str, default=None,
                        help='Path to aiXcoder weights .pt file (e.g., aix3-7b-base.pt)')
    parser.add_argument('--hf-cache-dir', type=str, default=None,
                        help='Path to HuggingFace cache directory for DeepSeek/CodeLlama')
    parser.add_argument('--output-dir', type=str, default='no_finetune_results',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation (default: 32)')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                        help='Random seeds for LSTM initialization (default: 1-20)')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['aixcoder', 'deepseek-coder', 'codellama'],
                        choices=['aixcoder', 'deepseek-coder', 'codellama'],
                        help='Models to evaluate (default: all three)')
    parser.add_argument('--no-freeze', action='store_true',
                        help='Do NOT freeze embedding layer (default: freeze)')
    
    args = parser.parse_args()
    
    # Validation
    if 'aixcoder' in args.models and args.aixcoder_weights is None:
        parser.error("--aixcoder-weights required when evaluating aiXcoder")
    if any(m in args.models for m in ['deepseek-coder', 'codellama']) and args.hf_cache_dir is None:
        parser.error("--hf-cache-dir required when evaluating DeepSeek or CodeLlama")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Config
    config = DEFAULT_CONFIG.copy()
    config['batch_size'] = args.batch_size
    freeze = not args.no_freeze
    
    # Load Devign test data
    print(f"\nLoading Devign test data from: {args.devign_dir}")
    test_loader = load_dataset(args.devign_dir, split='test', batch_size=args.batch_size)
    
    dataset_info = {
        'name': 'Devign',
        'path': args.devign_dir,
        'split': 'test',
    }
    
    # Run evaluations
    start_time = time.time()
    all_results = []
    
    # Model display names
    model_display_names = {
        'aixcoder': 'aiXcoder-7B',
        'deepseek-coder': 'DeepSeek-Coder-6.7B',
        'codellama': 'CodeLlama-7B',
    }
    
    for model_name in args.models:
        print(f"\n{'#'*60}")
        print(f"# Loading embeddings: {model_display_names[model_name]}")
        print(f"{'#'*60}")
        
        try:
            if model_name == 'aixcoder':
                emb_weights, vocab_size, emb_dim = load_aixcoder_embeddings(args.aixcoder_weights)
            else:
                emb_weights, vocab_size, emb_dim = load_hf_embeddings(args.hf_cache_dir, model_name)
            
            result = run_no_finetune_evaluation(
                model_label=model_display_names[model_name],
                embedding_weights=emb_weights,
                vocab_size=vocab_size,
                embedding_dim=emb_dim,
                test_loader=test_loader,
                device=device,
                config=config,
                seeds=args.seeds,
                freeze_embeddings=freeze,
            )
            all_results.append(result)
            
            # Free embedding weights from memory
            del emb_weights
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nERROR evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total evaluation time: {timedelta(seconds=int(elapsed))}")
    print(f"{'='*60}")
    
    if not all_results:
        print("ERROR: No models were successfully evaluated!")
        return
    
    # Save results
    save_results(all_results, args.output_dir, dataset_info)
    
    # Final comparison table
    print(f"\n{'='*70}")
    print(f"{'FINAL COMPARISON':^70}")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'AUROC':>12} {'F1':>12} {'Accuracy':>12} {'Precision':>12} {'Recall':>12}")
    print(f"{'-'*70}")
    for r in all_results:
        agg = r['aggregate']
        print(f"{r['model']:<25} "
              f"{agg['auroc_mean']:.4f}±{agg['auroc_std']:.4f} "
              f"{agg['f1_mean']:.4f}±{agg['f1_std']:.4f} "
              f"{agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f} "
              f"{agg['precision_mean']:.4f}±{agg['precision_std']:.4f} "
              f"{agg['recall_mean']:.4f}±{agg['recall_std']:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
