#!/usr/bin/env python3
"""
Memory monitoring and optimization helper for the ultrasound report generation model.
"""

import torch
import gc
import os

def get_gpu_memory_info():
    """Get GPU memory information"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
        allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # GB
        reserved_memory = torch.cuda.memory_reserved(device) / (1024**3)  # GB
        free_memory = total_memory - reserved_memory
        
        print(f"GPU {device} Memory Info:")
        print(f"  Total: {total_memory:.2f} GB")
        print(f"  Allocated: {allocated_memory:.2f} GB")
        print(f"  Reserved: {reserved_memory:.2f} GB")
        print(f"  Free: {free_memory:.2f} GB")
        
        return {
            'total': total_memory,
            'allocated': allocated_memory,
            'reserved': reserved_memory,
            'free': free_memory
        }
    else:
        print("CUDA not available")
        return None

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        print("Clearing GPU memory cache...")
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU memory cache cleared.")

def set_memory_optimization():
    """Set environment variables for memory optimization"""
    # Enable expandable segments to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Set other memory optimizations
    torch.backends.cudnn.benchmark = False  # Disable cudnn benchmark for consistent memory usage
    torch.backends.cudnn.deterministic = True
    
    print("Memory optimization settings applied:")
    print("  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    print("  - cudnn.benchmark=False")
    print("  - cudnn.deterministic=True")

def get_recommended_batch_size(available_memory_gb):
    """Get recommended batch size based on available GPU memory"""
    if available_memory_gb >= 24:
        return 32
    elif available_memory_gb >= 16:
        return 16
    elif available_memory_gb >= 12:
        return 12
    elif available_memory_gb >= 8:
        return 8
    elif available_memory_gb >= 6:
        return 4
    else:
        return 2

if __name__ == "__main__":
    print("=== GPU Memory Monitor ===")
    set_memory_optimization()
    print()
    
    memory_info = get_gpu_memory_info()
    if memory_info:
        recommended_batch = get_recommended_batch_size(memory_info['total'])
        print(f"\nRecommended batch size: {recommended_batch}")
        
        print("\nMemory saving tips:")
        print("1. Reduce batch_size in config.py")
        print("2. Reduce max_seq_length in config.py") 
        print("3. Reduce num_layers in config.py")
        print("4. Use gradient accumulation instead of large batch size")
        print("5. Enable mixed precision training (fp16)")
    
    print("\nTo use this in your training script, add:")
    print("import memory_monitor")
    print("memory_monitor.set_memory_optimization()")
    print("memory_monitor.clear_gpu_memory()  # Call when OOM occurs")
