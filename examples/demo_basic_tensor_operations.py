#!/usr/bin/env python

"""
Demo for basic tensor operations with TorchDevice.

This script demonstrates how basic tensor operations are handled 
by TorchDevice with seamless device redirection.
"""
# Import TorchDevice first to ensure it patches PyTorch
import TorchDevice  # noqa: F401
import torch
from example_utils import set_deterministic_seed

# Disable PyTorch compiler (torch._dynamo) which is causing issues
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

def main():
    print("\n=== Testing TorchDevice Basic Tensor Operations ===\n")
    
    # Set deterministic seed for reproducible results
    set_deterministic_seed()
    
    # Device automatically redirects to the best available hardware
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Create tensors with random values
    print("\nCreating tensors on device...")
    a = torch.randn(3, 3).to(device)
    b = torch.randn(3, 3).to(device)
    print(f"Tensor a on {a.device}:")
    print(a)
    print(f"Tensor b on {b.device}:")
    print(b)
    
    # Basic arithmetic operations
    print("\nPerforming basic operations...")
    c = a + b
    print("\nAddition (a + b):")
    print(c)
    
    d = a * b
    print("\nElement-wise multiplication (a * b):")
    print(d)
    
    e = torch.matmul(a, b)
    print("\nMatrix multiplication (torch.matmul(a, b)):")
    print(e)
    
    # Other common operations
    f = torch.sin(a)
    print("\nSine of a (torch.sin(a)):")
    print(f)
    
    g = torch.exp(b)
    print("\nExponential of b (torch.exp(b)):")
    print(g)
    
    h = torch.max(a, b)
    print("\nElement-wise maximum (torch.max(a, b)):")
    print(h)
    
    # Change device if needed
    cpu_tensor = a.cpu()
    print(f"\nTensor moved to CPU: {cpu_tensor.device}")
    
    # Move back to accelerator
    accelerated_tensor = cpu_tensor.to(device)
    print(f"Tensor moved back to accelerator: {accelerated_tensor.device}")
    
    # Test device properties
    print("\nTesting device properties...")
    print(f"Device name: {torch.cuda.get_device_name(device)}")
    print(f"Device memory allocated: {torch.cuda.memory_allocated(device)}")
    print(f"Device memory reserved: {torch.cuda.memory_reserved(device)}")
    
    print("\n=== Test Complete ===")

if __name__ == '__main__':
    main()