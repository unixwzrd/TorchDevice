#!/usr/bin/env python3
"""
Demo of stream functions with TorchDevice.
This demo tests each stream function individually with proper error handling.
"""

import os
import sys

# Add the parent directory to the path so we can import TorchDevice
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from TorchDevice import TorchDevice

# Apply patches to make CUDA functions work on MPS
TorchDevice.apply_patches()

def test_function(name, func):
    """Test a function and handle any exceptions."""
    print(f"Testing {name}...")
    try:
        result = func()
        print(f"✓ {name} succeeded: {result}")
        return result
    except Exception as e:
        print(f"✗ {name} failed: {e}")
        return None

def main():
    print("Stream Functions Demo")
    print("====================")
    
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
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)
    
    print("\n1. Basic Stream Creation")
    print("----------------------")
    
    # Test creating a stream
    s1 = test_function("torch.cuda.stream()", lambda: torch.cuda.stream())
    
    # Test getting the default stream
    default_stream = test_function("torch.cuda.default_stream()", 
                                  lambda: torch.cuda.default_stream())
    
    # Test getting the current stream
    current_stream = test_function("torch.cuda.current_stream()", 
                                  lambda: torch.cuda.current_stream())
    
    print("\n2. Stream Properties")
    print("------------------")
    
    if s1:
        # Test stream query
        test_function("stream.query()", lambda: s1.query())
        
        # Test stream priority (may not be supported)
        test_function("stream.priority", lambda: getattr(s1, 'priority', None))
    
    print("\n3. Event Creation")
    print("---------------")
    
    # Test creating an event
    e1 = test_function("torch.cuda.Event()", lambda: torch.cuda.Event())
    
    # Test creating an event with timing
    e_timing = test_function("torch.cuda.Event(enable_timing=True)", 
                           lambda: torch.cuda.Event(enable_timing=True))
    
    print("\n4. Event Recording")
    print("----------------")
    
    if e1 and s1:
        # Test recording an event
        test_function("event.record()", lambda: e1.record())
        
        # Test recording an event on a specific stream
        test_function("event.record(stream)", lambda: e_timing.record(s1))
    
    print("\n5. Stream Methods")
    print("---------------")
    
    if s1:
        # Test stream record_event
        e2 = test_function("stream.record_event()", lambda: s1.record_event())
    
    print("\n6. General torch.Stream API")
    print("-------------------------")
    
    if hasattr(torch, 'Stream'):
        # Test creating a Stream
        general_stream = test_function("torch.Stream(device=device)", 
                                     lambda: torch.Stream(device=device))
        
        if general_stream:
            # Test recording an event on the general stream
            test_function("general_stream.record_event()", 
                        lambda: general_stream.record_event())
    else:
        print("torch.Stream is not available in this PyTorch version")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
