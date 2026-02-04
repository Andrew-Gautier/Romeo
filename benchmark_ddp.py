"""
GPU Benchmark Script comparing DataParallel vs DDP
Run with: torchrun --standalone --nproc_per_node=8 benchmark_ddp.py [args]
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import time
import argparse
from tqdm import tqdm

from binary_classifier import LSTMClassifier, create_model


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
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size():
    """Get number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def print_rank0(*args, **kwargs):
    """Print only from rank 0."""
    if is_main_process():
        print(*args, **kwargs)


# ============================================================================
# Benchmark Functions
# ============================================================================

def load_data(data_dir, batch_size, use_distributed=False):
    """Load training data with optional distributed sampler."""
    sequences = torch.load(os.path.join(data_dir, 'train_sequences.pt')).long()
    labels = torch.load(os.path.join(data_dir, 'train_labels.pt'))
    
    dataset = TensorDataset(sequences, labels)
    
    if use_distributed and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    return loader, sampler


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


def benchmark_ddp(model, data_loader, sampler, device, num_batches, gradient_clip=1.0, warmup_batches=5):
    """Benchmark DDP training throughput."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss().to(device)
    
    batch_times = []
    samples_processed = 0
    
    # Set epoch for sampler
    if sampler is not None:
        sampler.set_epoch(0)
    
    # Warmup
    print_rank0(f"  Warming up ({warmup_batches} batches)...")
    data_iter = iter(data_loader)
    for i in range(warmup_batches):
        try:
            batch_sequences, batch_labels = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch_sequences, batch_labels = next(data_iter)
        
        batch_sequences = batch_sequences.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True).float()
        
        optimizer.zero_grad()
        predictions = model(batch_sequences).squeeze(1)
        loss = criterion(predictions, batch_labels)
        loss.backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
    
    # Synchronize before timing
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    
    print_rank0(f"  Benchmarking ({num_batches} batches)...")
    benchmark_start = time.time()
    
    iterator = range(num_batches)
    if is_main_process():
        iterator = tqdm(iterator, desc="  Batches", leave=False)
    
    for i in iterator:
        batch_start = time.time()
        
        try:
            batch_sequences, batch_labels = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch_sequences, batch_labels = next(data_iter)
        
        batch_sequences = batch_sequences.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True).float()
        
        optimizer.zero_grad()
        predictions = model(batch_sequences).squeeze(1)
        loss = criterion(predictions, batch_labels)
        loss.backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        torch.cuda.synchronize()
        batch_times.append(time.time() - batch_start)
        samples_processed += batch_sequences.size(0)
    
    # Synchronize at end
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    
    benchmark_elapsed = time.time() - benchmark_start
    
    # Aggregate samples across all ranks
    if dist.is_initialized():
        total_samples = torch.tensor([samples_processed], device=device)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        samples_processed = total_samples.item()
    
    return {
        'total_time': benchmark_elapsed,
        'samples_processed': samples_processed,
        'samples_per_sec': samples_processed / benchmark_elapsed,
        'avg_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark DDP training')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-batches', type=int, default=100)
    
    args = parser.parse_args()
    
    # Setup DDP
    local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
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
    
    try:
        print_rank0("="*70)
        print_rank0("DDP TRAINING BENCHMARK")
        print_rank0("="*70)
        print_rank0(f"\nConfiguration:")
        print_rank0(f"  World size: {get_world_size()} GPUs")
        print_rank0(f"  Batch size per GPU: {args.batch_size}")
        print_rank0(f"  Effective batch size: {args.batch_size * get_world_size()}")
        print_rank0(f"  Number of batches: {args.num_batches}")
        
        if is_main_process():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"    GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        
        # Load weights
        print_rank0(f"\nLoading pretrained weights...")
        pretrained = torch.load(args.weights, map_location='cpu')
        pretrained_weights = pretrained['tok_embeddings.weight']
        
        # Load data with distributed sampler
        print_rank0(f"\nLoading data...")
        data_loader, sampler = load_data(args.data_dir, args.batch_size, use_distributed=True)
        print_rank0(f"Batches per GPU: {len(data_loader)}")
        
        # Create model and wrap with DDP
        print_rank0(f"\nCreating model with DDP...")
        model = create_test_model(config, pretrained_weights, device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
        num_params = sum(p.numel() for p in model.parameters())
        print_rank0(f"Model parameters: {num_params:,}")
        
        # Run benchmark
        print_rank0(f"\n" + "="*70)
        print_rank0(f"Running DDP Benchmark ({get_world_size()} GPUs)")
        print_rank0("="*70)
        
        results = benchmark_ddp(model, data_loader, sampler, device, args.num_batches)
        
        print_rank0(f"\n  Results:")
        print_rank0(f"    Total time: {results['total_time']:.2f}s")
        print_rank0(f"    Total samples processed: {results['samples_processed']}")
        print_rank0(f"    Throughput: {results['samples_per_sec']:.2f} samples/sec")
        print_rank0(f"    Avg batch time: {results['avg_batch_time']*1000:.2f}ms ± {results['std_batch_time']*1000:.2f}ms")
        
        # Print comparison with expected single-GPU baseline
        # Your single GPU with batch_size=16 got ~27.76 samples/sec
        single_gpu_baseline = 27.76  # From your previous benchmark
        expected_linear = single_gpu_baseline * get_world_size()
        actual_speedup = results['samples_per_sec'] / single_gpu_baseline
        efficiency = actual_speedup / get_world_size() * 100
        
        print_rank0(f"\n  Comparison with DataParallel baseline:")
        print_rank0(f"    Single GPU baseline: ~{single_gpu_baseline:.1f} samples/sec")
        print_rank0(f"    Ideal linear scaling: {expected_linear:.1f} samples/sec")
        print_rank0(f"    DDP actual: {results['samples_per_sec']:.1f} samples/sec")
        print_rank0(f"    Speedup vs single GPU: {actual_speedup:.2f}x")
        print_rank0(f"    Scaling efficiency: {efficiency:.1f}%")
        
        if efficiency >= 70:
            print_rank0(f"\n✓ DDP scaling efficiency is good!")
        elif efficiency >= 50:
            print_rank0(f"\n⚠️ DDP scaling efficiency is moderate. Consider larger batch sizes.")
        else:
            print_rank0(f"\n⚠️ DDP scaling efficiency is low. Check for bottlenecks.")
        
        # Compare with DataParallel result
        dataparallel_throughput = 40.35  # From your benchmark
        improvement = results['samples_per_sec'] / dataparallel_throughput
        print_rank0(f"\n  Improvement over DataParallel:")
        print_rank0(f"    DataParallel was: {dataparallel_throughput:.1f} samples/sec")
        print_rank0(f"    DDP is: {results['samples_per_sec']:.1f} samples/sec")
        print_rank0(f"    Improvement: {improvement:.2f}x faster")
        
        print_rank0("\n" + "="*70)
        print_rank0("BENCHMARK COMPLETE")
        print_rank0("="*70)
        
    finally:
        cleanup_ddp()


if __name__ == '__main__':
    main()
