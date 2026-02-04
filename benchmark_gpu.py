"""
GPU Benchmark Script for LSTM Training
Compares single-GPU vs multi-GPU (DataParallel) performance.

This script helps diagnose performance issues with multi-GPU training by:
1. Running a fixed number of training batches on a single GPU
2. Running the same batches on multiple GPUs with DataParallel
3. Comparing throughput (samples/sec) and timing breakdowns
"""

import torch
import torch.nn as nn
import numpy as np
import os
import time
import argparse
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from binary_classifier import LSTMClassifier, create_model


def get_gpu_memory_info():
    """Get memory info for all GPUs."""
    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = props.total_memory / 1024**3
        info.append({
            'gpu': i,
            'name': props.name,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - reserved
        })
    return info


def print_gpu_memory():
    """Print current GPU memory usage."""
    print("\n" + "="*70)
    print("GPU Memory Status")
    print("="*70)
    for info in get_gpu_memory_info():
        print(f"  GPU {info['gpu']}: {info['name']}")
        print(f"    Allocated: {info['allocated_gb']:.2f} GB, Reserved: {info['reserved_gb']:.2f} GB, Free: {info['free_gb']:.2f} GB")
    print("="*70 + "\n")


def load_data(data_dir, batch_size):
    """Load training data."""
    sequences = torch.load(os.path.join(data_dir, 'train_sequences.pt')).long()
    labels = torch.load(os.path.join(data_dir, 'train_labels.pt'))
    
    print(f"Data shape: sequences={sequences.shape}, labels={labels.shape}")
    
    dataset = TensorDataset(sequences, labels)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )
    
    return loader


def create_test_model(config, pretrained_weights, device):
    """Create model for benchmarking."""
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
    return model


def benchmark_training(model, data_loader, device, num_batches, gradient_clip=1.0, warmup_batches=5):
    """
    Benchmark training throughput.
    
    Returns dict with timing stats.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss().to(device)
    
    # Timing accumulators
    data_load_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    total_times = []
    samples_processed = 0
    
    # Warmup phase (not counted)
    print(f"  Warming up ({warmup_batches} batches)...")
    data_iter = iter(data_loader)
    for i in range(warmup_batches):
        try:
            batch_sequences, batch_labels = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch_sequences, batch_labels = next(data_iter)
        
        batch_sequences = batch_sequences.to(device)
        batch_labels = batch_labels.to(device).float()
        
        optimizer.zero_grad()
        predictions = model(batch_sequences).squeeze(1)
        loss = criterion(predictions, batch_labels)
        loss.backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    print(f"  Benchmarking ({num_batches} batches)...")
    benchmark_start = time.time()
    
    for i in tqdm(range(num_batches), desc="  Batches", leave=False):
        batch_start = time.time()
        
        # Data loading
        t0 = time.time()
        try:
            batch_sequences, batch_labels = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch_sequences, batch_labels = next(data_iter)
        
        batch_sequences = batch_sequences.to(device)
        batch_labels = batch_labels.to(device).float()
        torch.cuda.synchronize()
        data_load_times.append(time.time() - t0)
        
        # Forward pass
        t0 = time.time()
        optimizer.zero_grad()
        predictions = model(batch_sequences).squeeze(1)
        loss = criterion(predictions, batch_labels)
        torch.cuda.synchronize()
        forward_times.append(time.time() - t0)
        
        # Backward pass
        t0 = time.time()
        loss.backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        torch.cuda.synchronize()
        backward_times.append(time.time() - t0)
        
        # Optimizer step
        t0 = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        optimizer_times.append(time.time() - t0)
        
        total_times.append(time.time() - batch_start)
        samples_processed += batch_sequences.size(0)
    
    torch.cuda.synchronize()
    benchmark_elapsed = time.time() - benchmark_start
    
    return {
        'total_time': benchmark_elapsed,
        'samples_processed': samples_processed,
        'samples_per_sec': samples_processed / benchmark_elapsed,
        'avg_batch_time': np.mean(total_times),
        'avg_data_load_time': np.mean(data_load_times),
        'avg_forward_time': np.mean(forward_times),
        'avg_backward_time': np.mean(backward_times),
        'avg_optimizer_time': np.mean(optimizer_times),
        'std_batch_time': np.std(total_times),
    }


def run_single_gpu_benchmark(data_loader, pretrained_weights, config, num_batches, gpu_id=0):
    """Run benchmark on a single GPU."""
    print("\n" + "="*70)
    print(f"SINGLE GPU BENCHMARK (GPU {gpu_id})")
    print("="*70)
    
    device = torch.device(f'cuda:{gpu_id}')
    print(f"Using device: {device}")
    
    # Create model
    model = create_test_model(config, pretrained_weights, device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    print_gpu_memory()
    
    # Run benchmark
    results = benchmark_training(model, data_loader, device, num_batches)
    
    print(f"\n  Results:")
    print(f"    Total time: {results['total_time']:.2f}s")
    print(f"    Samples processed: {results['samples_processed']}")
    print(f"    Throughput: {results['samples_per_sec']:.2f} samples/sec")
    print(f"    Avg batch time: {results['avg_batch_time']*1000:.2f}ms ± {results['std_batch_time']*1000:.2f}ms")
    print(f"    Breakdown:")
    print(f"      Data loading: {results['avg_data_load_time']*1000:.2f}ms ({100*results['avg_data_load_time']/results['avg_batch_time']:.1f}%)")
    print(f"      Forward pass: {results['avg_forward_time']*1000:.2f}ms ({100*results['avg_forward_time']/results['avg_batch_time']:.1f}%)")
    print(f"      Backward pass: {results['avg_backward_time']*1000:.2f}ms ({100*results['avg_backward_time']/results['avg_batch_time']:.1f}%)")
    print(f"      Optimizer step: {results['avg_optimizer_time']*1000:.2f}ms ({100*results['avg_optimizer_time']/results['avg_batch_time']:.1f}%)")
    
    print_gpu_memory()
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results


def run_multi_gpu_benchmark(data_loader, pretrained_weights, config, num_batches, num_gpus=None):
    """Run benchmark with DataParallel on multiple GPUs."""
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    print("\n" + "="*70)
    print(f"MULTI-GPU BENCHMARK (DataParallel with {num_gpus} GPUs)")
    print("="*70)
    
    device = torch.device('cuda:0')
    print(f"Primary device: {device}")
    
    # Create model on primary device
    model = create_test_model(config, pretrained_weights, device)
    
    # Wrap in DataParallel
    if num_gpus > 1:
        gpu_ids = list(range(num_gpus))
        print(f"Using GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    print_gpu_memory()
    
    # Run benchmark
    results = benchmark_training(model, data_loader, device, num_batches)
    
    print(f"\n  Results:")
    print(f"    Total time: {results['total_time']:.2f}s")
    print(f"    Samples processed: {results['samples_processed']}")
    print(f"    Throughput: {results['samples_per_sec']:.2f} samples/sec")
    print(f"    Avg batch time: {results['avg_batch_time']*1000:.2f}ms ± {results['std_batch_time']*1000:.2f}ms")
    print(f"    Breakdown:")
    print(f"      Data loading: {results['avg_data_load_time']*1000:.2f}ms ({100*results['avg_data_load_time']/results['avg_batch_time']:.1f}%)")
    print(f"      Forward pass: {results['avg_forward_time']*1000:.2f}ms ({100*results['avg_forward_time']/results['avg_batch_time']:.1f}%)")
    print(f"      Backward pass: {results['avg_backward_time']*1000:.2f}ms ({100*results['avg_backward_time']/results['avg_batch_time']:.1f}%)")
    print(f"      Optimizer step: {results['avg_optimizer_time']*1000:.2f}ms ({100*results['avg_optimizer_time']/results['avg_batch_time']:.1f}%)")
    
    print_gpu_memory()
    
    # Cleanup
    del model
    for i in range(num_gpus):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark GPU training performance')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory (e.g., juliet_c_simhash_k=1_...)')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to pretrained embedding weights')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size per GPU (default: 8)')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='Number of batches to benchmark (default: 100)')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs for multi-GPU test (default: all available)')
    parser.add_argument('--single-only', action='store_true',
                        help='Only run single GPU benchmark')
    parser.add_argument('--multi-only', action='store_true',
                        help='Only run multi-GPU benchmark')
    
    args = parser.parse_args()
    
    # Configuration (same as training)
    config = {
        'batch_size': args.batch_size,
        'learning_rate': 0.001,
        'lstm_nodes': 256,
        'vocab_size': 49152,
        'embedding_size': 4096,
        'output_dim': 1,
        'n_layers': 2,
        'bidirectional': True,
        'dropout': 0.5,
        'n_heads': 8,
    }
    
    print("="*70)
    print("GPU TRAINING BENCHMARK")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Weights path: {args.weights}")
    print(f"  Batch size (per GPU): {args.batch_size}")
    print(f"  Number of batches: {args.num_batches}")
    print(f"  Available GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"    GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    # Load pretrained weights
    print(f"\nLoading pretrained weights from {args.weights}...")
    pretrained = torch.load(args.weights, map_location='cpu')
    pretrained_weights = pretrained['tok_embeddings.weight']
    print(f"Pretrained weights shape: {pretrained_weights.shape}")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    
    # For multi-GPU benchmark, we want the same effective batch size
    # So if single GPU uses batch_size=8, multi-GPU with 8 GPUs uses batch_size=8 per GPU = 64 total
    single_gpu_batch_size = args.batch_size
    multi_gpu_batch_size = args.batch_size  # Per GPU, so effective = batch_size * num_gpus
    
    single_loader = load_data(args.data_dir, single_gpu_batch_size)
    multi_loader = load_data(args.data_dir, multi_gpu_batch_size)
    
    print(f"Single GPU loader: {len(single_loader)} batches of {single_gpu_batch_size}")
    print(f"Multi GPU loader: {len(multi_loader)} batches of {multi_gpu_batch_size}")
    
    results = {}
    
    # Run single GPU benchmark
    if not args.multi_only:
        results['single_gpu'] = run_single_gpu_benchmark(
            single_loader, pretrained_weights, config, args.num_batches
        )
    
    # Run multi-GPU benchmark
    if not args.single_only and torch.cuda.device_count() > 1:
        num_gpus = args.num_gpus or torch.cuda.device_count()
        results['multi_gpu'] = run_multi_gpu_benchmark(
            multi_loader, pretrained_weights, config, args.num_batches, num_gpus
        )
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    if 'single_gpu' in results:
        print(f"\nSingle GPU (batch_size={single_gpu_batch_size}):")
        print(f"  Throughput: {results['single_gpu']['samples_per_sec']:.2f} samples/sec")
        print(f"  Avg batch time: {results['single_gpu']['avg_batch_time']*1000:.2f}ms")
    
    if 'multi_gpu' in results:
        num_gpus = args.num_gpus or torch.cuda.device_count()
        effective_batch = multi_gpu_batch_size * num_gpus
        print(f"\nMulti-GPU ({num_gpus} GPUs, batch_size={multi_gpu_batch_size}/GPU, effective={effective_batch}):")
        print(f"  Throughput: {results['multi_gpu']['samples_per_sec']:.2f} samples/sec")
        print(f"  Avg batch time: {results['multi_gpu']['avg_batch_time']*1000:.2f}ms")
    
    if 'single_gpu' in results and 'multi_gpu' in results:
        speedup = results['multi_gpu']['samples_per_sec'] / results['single_gpu']['samples_per_sec']
        num_gpus = args.num_gpus or torch.cuda.device_count()
        efficiency = speedup / num_gpus * 100
        
        print(f"\nSpeedup Analysis:")
        print(f"  Actual speedup: {speedup:.2f}x")
        print(f"  Ideal speedup (linear): {num_gpus}x")
        print(f"  Scaling efficiency: {efficiency:.1f}%")
        
        if efficiency < 50:
            print("\n⚠️  WARNING: Scaling efficiency is low!")
            print("   Possible causes:")
            print("   - DataParallel has high communication overhead")
            print("   - Batch size may be too small per GPU")
            print("   - Model may be I/O bound rather than compute bound")
            print("   - Consider using DistributedDataParallel (DDP) instead")
        elif efficiency < 75:
            print("\n⚠️  Note: Scaling efficiency is moderate.")
            print("   Consider trying DistributedDataParallel (DDP) for better performance.")
        else:
            print("\n✓ Scaling efficiency is good.")
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
