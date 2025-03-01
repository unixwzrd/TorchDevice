#!/usr/bin/env python3
"""
Demo of stream operations with TorchDevice.

This example demonstrates how to use CUDA streams with TorchDevice,
which will redirect to MPS streams when running on Apple Silicon.
"""

import os
import sys
import time

# Add the parent directory to the path so we can import TorchDevice
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from TorchDevice import TorchDevice

# Apply patches to make CUDA functions work on MPS
TorchDevice.apply_patches()

def main():
    print("Stream Operations Demo")
    print("=====================")
    
    # Check if CUDA is available (will return True on MPS devices)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Create a tensor on the GPU
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    
    print("\nBasic Stream Operations:")
    print("-----------------------")
    
    # Get the default stream
    default_stream = torch.cuda.default_stream()
    print(f"Default stream created")
    
    # Create a new stream
    s1 = torch.cuda.stream()
    print(f"New stream created")
    
    # Check if operations are complete
    print(f"Default stream query: {default_stream.query()}")
    print(f"New stream query: {s1.query()}")
    
    # Synchronize streams
    default_stream.synchronize()
    s1.synchronize()
    print("Streams synchronized")
    
    print("\nEvent Operations:")
    print("----------------")
    
    # Create an event
    e1 = torch.cuda.Event(enable_timing=True)
    print("Event created with timing enabled")
    
    # Record the event
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Record the start event
    start_event.record()
    
    # Do some computation
    z = torch.matmul(x, y)
    
    # Record the end event
    end_event.record()
    
    # Synchronize to get accurate timing
    end_event.synchronize()
    
    # Calculate elapsed time
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"Matrix multiplication took {elapsed_time:.2f} ms")
    
    print("\nStream Synchronization:")
    print("----------------------")
    
    # Create two streams
    s1 = torch.cuda.stream()
    s2 = torch.cuda.stream()
    
    # Record an event in s1
    e1 = s1.record_event()
    
    # Make s2 wait for e1
    s2.wait_event(e1)
    print("Stream 2 waiting for event from Stream 1")
    
    # Make one stream wait for another
    s2.wait_stream(s1)
    print("Stream 2 waiting for Stream 1")
    
    print("\nGeneral Stream API:")
    print("-----------------")
    
    # Using the general torch.Stream API
    if hasattr(torch, 'Stream'):
        s = torch.Stream(device='cuda')
        print(f"Created a general Stream for device: cuda")
        
        # Record an event
        e = s.record_event()
        print("Event recorded in the stream")
        
        # Synchronize the stream
        s.synchronize()
        print("Stream synchronized")
    else:
        print("torch.Stream is not available in this PyTorch version")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
