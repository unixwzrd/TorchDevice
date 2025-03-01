#!/usr/bin/env python

# Demo: Basic Tensor Operations with TorchDevice

import os
# Disable PyTorch compiler (torch._dynamo) which is causing issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import TorchDevice
import torch

def main():
    print("\n=== Testing TorchDevice Basic Tensor Operations ===\n")
    
    # Create a device using CUDA syntax (will be redirected if on MPS)
    print("Creating device...")
    device = torch.device('cuda')
    print(f"Device created: {device}")
    
    # Create tensors on the device
    print("\nCreating tensors on device...")
    a = torch.randn(3, 3).to(device)
    b = torch.randn(3, 3).to(device)
    print(f"Tensor a: {a}")
    print(f"Tensor b: {b}")
    
    # Perform basic operations
    print("\nPerforming basic operations...")
    c = a + b
    print(f"Addition (a + b): {c}")
    
    d = a @ b  # Matrix multiplication
    print(f"Matrix multiplication (a @ b): {d}")
    
    e = torch.sin(a)
    print(f"Sine function (sin(a)): {e}")
    
    # Test device properties
    print("\nDevice properties:")
    if hasattr(torch.cuda, 'get_device_name'):
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    if hasattr(torch.cuda, 'memory_allocated'):
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
    
    if hasattr(torch.cuda, 'memory_reserved'):
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
    
    print("\n=== Test Complete ===")

if __name__ == '__main__':
    main()