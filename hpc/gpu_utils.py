"""
GPU utility functions for robust GPU selection and memory management.
"""

import os
import torch
import torch.cuda as cuda
import time

def print_gpu_status():
    """Print status of all GPUs."""
    if not cuda.is_available():
        print("CUDA not available")
        return
    
    print("\n" + "="*60)
    print("GPU Status")
    print("="*60)
    
    for i in range(cuda.device_count()):
        try:
            torch.cuda.set_device(i)
            props = cuda.get_device_properties(i)
            
            allocated = cuda.memory_allocated(i) / 1024**3
            cached = cuda.memory_reserved(i) / 1024**3
            total = props.total_memory / 1024**3
            free = total - (allocated + cached)
            
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {free:.1f} GB free / {total:.1f} GB total")
            print(f"  Usage: {allocated:.1f} GB allocated, {cached:.1f} GB cached")
            
        except Exception as e:
            print(f"GPU {i}: Error - {e}")
    
    print("="*60 + "\n")

def select_best_gpu(min_free_gb=15):
    """
    Select the best available GPU based on free memory.
    
    Args:
        min_free_gb: Minimum free GB required
        
    Returns:
        torch.device: Selected device
    """
    if not cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')
    
    print_gpu_status()
    
    # Check each GPU
    candidate_gpus = []
    
    for i in range(cuda.device_count()):
        try:
            torch.cuda.set_device(i)
            
            props = cuda.get_device_properties(i)
            total_gb = props.total_memory / 1024**3
            allocated_gb = cuda.memory_allocated(i) / 1024**3
            cached_gb = cuda.memory_reserved(i) / 1024**3
            free_gb = total_gb - (allocated_gb + cached_gb)
            
            print(f"GPU {i}: {free_gb:.1f}/{total_gb:.1f} GB free")
            
            if free_gb >= min_free_gb:
                candidate_gpus.append((i, free_gb, total_gb))
                
        except Exception as e:
            print(f"GPU {i} error: {e}")
            continue
    
    # Select best GPU
    if candidate_gpus:
        # Sort by free memory (descending)
        candidate_gpus.sort(key=lambda x: x[1], reverse=True)
        selected_gpu = candidate_gpus[0][0]
        free_gb = candidate_gpus[0][1]
        total_gb = candidate_gpus[0][2]
        
        print(f"\n✓ Selected GPU {selected_gpu} ({free_gb:.1f}/{total_gb:.1f} GB free)")
        
        # Set environment variable to isolate this GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        
        # Clear cache on selected GPU
        torch.cuda.set_device(selected_gpu)
        cuda.empty_cache()
        time.sleep(1)
        
        return torch.device('cuda:0')
    
    # If no GPU meets threshold, try to find any with low usage
    print(f"\nNo GPU with {min_free_gb} GB free, looking for lightly used GPU...")
    
    for i in range(cuda.device_count()):
        try:
            torch.cuda.set_device(i)
            allocated_gb = cuda.memory_allocated(i) / 1024**3
            
            if allocated_gb < 0.5:  # Less than 0.5GB used
                print(f"Selected GPU {i} (low usage: {allocated_gb:.1f} GB allocated)")
                os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
                return torch.device('cuda:0')
                
        except Exception as e:
            print(f"GPU {i} error: {e}")
            continue
    
    print("⚠ No suitable GPU found, using CPU")
    return torch.device('cpu')

def clear_gpu_memory(verbose=True):
    """
    Aggressively clear GPU memory cache on all devices.
    Call this between training runs to prevent memory buildup.
    
    Args:
        verbose: Whether to print memory status
    """
    if not cuda.is_available():
        return
    
    import gc
    
    # Run garbage collection first
    gc.collect()
    
    # Clear cache on all GPUs
    for i in range(cuda.device_count()):
        try:
            with torch.cuda.device(i):
                cuda.empty_cache()
                cuda.synchronize()
        except Exception as e:
            if verbose:
                print(f"Warning: Could not clear GPU {i}: {e}")
    
    # Small delay to ensure memory is released
    time.sleep(0.5)
    
    if verbose:
        print("GPU memory cleared on all devices")

def limit_gpu_memory(fraction=0.8):
    """
    Limit GPU memory usage to prevent OOM.
    
    Args:
        fraction: Fraction of total memory to use (0.0 to 1.0)
    """
    if cuda.is_available():
        for i in range(cuda.device_count()):
            try:
                torch.cuda.set_per_process_memory_fraction(fraction, device=i)
                print(f"GPU {i}: Limited to {fraction*100:.0f}% of memory")
            except:
                pass