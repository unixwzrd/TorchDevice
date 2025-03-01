#!/usr/bin/env python3
"""
Simple demo of TorchDevice with basic operations.
This demo avoids any stream synchronization or event handling that might cause hanging.
"""

import os
import sys

# Add the parent directory to the path so we can import TorchDevice
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from TorchDevice import TorchDevice

# Apply patches to make CUDA functions work on MPS
TorchDevice.apply_patches()

def main():
    print("Simple TorchDevice Demo")
    print("======================")
    
    # Display device information
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Get the appropriate device
    device = 'mps'
    print(f"Using device: {device}")
    
    # Create tensors on the GPU
    print("\nCreating tensors on device...")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    # Perform basic operations
    print("Performing basic operations...")
    z = x + y
    print(f"Addition shape: {z.shape}")
    
    z = torch.matmul(x, y)
    print(f"Matrix multiplication shape: {z.shape}")
    
    # Check device of tensors
    print(f"\nTensor x device: {x.device}")
    print(f"Tensor y device: {y.device}")
    print(f"Tensor z device: {z.device}")
    
    # Create a stream without synchronization
    print("\nCreating a stream (without synchronization)...")
    s = torch.cuda.stream()
    print("Stream created successfully")
    
    # Create an event without synchronization
    print("\nCreating an event (without synchronization)...")
    e = torch.cuda.Event()
    print("Event created successfully")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
