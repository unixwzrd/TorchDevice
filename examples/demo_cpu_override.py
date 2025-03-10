#!/usr/bin/env python
"""
Demo script showcasing the CPU override feature in TorchDevice.

This script demonstrates how to:
1. Use the special 'cpu:-1' syntax to force CPU usage
2. Verify that subsequent operations respect the CPU override
3. Compare performance between default device and forced CPU
"""
import time
import torch
import TorchDevice

def print_device_info(name, device, tensor):
    """Print information about a device and a tensor on that device."""
    print(f"[{name}]")
    print(f"  - Device: {device}")
    print(f"  - Tensor device: {tensor.device}")
    print(f"  - Default device: {TorchDevice.TorchDevice._default_device}")
    print(f"  - CPU override active: {TorchDevice.TorchDevice._cpu_override}")
    print()

def run_benchmark(device, size=1000):
    """Run a simple matrix multiplication benchmark on the specified device."""
    # Create matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warmup
    for _ in range(3):
        _ = torch.matmul(a, b)
    
    # Time the operation
    start_time = time.time()
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize() if hasattr(torch.cuda, "synchronize") else None
    torch.mps.synchronize() if hasattr(torch.mps, "synchronize") else None
    end_time = time.time()
    
    return end_time - start_time

# Part 1: Check the default device
print("=" * 80)
print("PART 1: Default Device Setup")
print("=" * 80)

# Create a tensor on the default device
default_device = torch.device("")  # Empty string gets the default device
default_tensor = torch.randn(5, 5)
print_device_info("Default", default_device, default_tensor)

# Part 2: Activate CPU override
print("=" * 80)
print("PART 2: Activating CPU Override")
print("=" * 80)

# Use the special 'cpu:-1' syntax to override to CPU
cpu_override_device = torch.device('cpu:-1')
cpu_override_tensor = torch.randn(5, 5)
print_device_info("After CPU Override", cpu_override_device, cpu_override_tensor)

# Part 3: Verify that subsequent operations respect the CPU override
print("=" * 80)
print("PART 3: Testing Subsequent Operations")
print("=" * 80)

# Try to create a tensor on CUDA/MPS
gpu_tensor = torch.randn(5, 5, device='cuda' if torch.cuda.is_available() else 'mps')
print_device_info("Attempted GPU Tensor", torch.device('cuda' if torch.cuda.is_available() else 'mps'), gpu_tensor)

# Create a neural network and move it to GPU
model = torch.nn.Sequential(
    torch.nn.Linear(10, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 1)
)
model_device = 'cuda' if torch.cuda.is_available() else 'mps'
model = model.to(model_device)
print(f"Model parameters device after .to({model_device}):")
for name, param in model.named_parameters():
    print(f"  - {name}: {param.device}")
print()

# Part 4: Performance comparison (small to avoid long run time)
print("=" * 80)
print("PART 4: Performance Impact")
print("=" * 80)

# Use a small matrix size to avoid long run times in examples
matrix_size = 100
cpu_time = run_benchmark('cpu', matrix_size)
print(f"Matrix multiplication ({matrix_size}x{matrix_size}) on CPU: {cpu_time:.6f} seconds")

print("\nNote: CPU override forces all operations to run on CPU regardless of available hardware.")
print("This feature is useful for debugging, testing, or when GPU memory is limited.")
print("\nTo reset the CPU override, you would need to restart your Python session.") 