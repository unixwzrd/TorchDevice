#!/usr/bin/env python3
"""
Demo of stream operations with TorchDevice.

This example demonstrates how to use CUDA streams with TorchDevice,
which will redirect to MPS streams when running on Apple Silicon.
"""

import os
import sys
import time
import signal

# Add the parent directory to the path so we can import TorchDevice
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from TorchDevice import TorchDevice

# Apply patches to make CUDA functions work on MPS
TorchDevice.apply_patches()

# Set up a timeout handler to prevent hanging
def timeout_handler(signum, frame):
    print("\nOperation timed out. This might be due to stream or event synchronization issues.")
    raise TimeoutError("Operation timed out")

def run_with_timeout(func, timeout=5):
    """Run a function with a timeout to prevent hanging."""
    # Set the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        result = func()
        # Cancel the alarm if the function completes
        signal.alarm(0)
        return result
    except TimeoutError as e:
        print(f"Timeout occurred: {e}")
        return None
    finally:
        # Ensure the alarm is canceled
        signal.alarm(0)

def main():
    print("Stream Operations Demo")
    print("=====================")
    
    # Check if CUDA is available (will return True on MPS devices)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Get the appropriate device
    device = 'mps'
    print(f"Using device: {device}")
    
    try:
        # Create tensors on the GPU
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
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
        
        # Synchronize streams with timeout
        def sync_default_stream():
            default_stream.synchronize()
            return "Default stream synchronized"
        
        def sync_s1_stream():
            s1.synchronize()
            return "Stream s1 synchronized"
        
        print(run_with_timeout(sync_default_stream) or "Default stream synchronization timed out")
        print(run_with_timeout(sync_s1_stream) or "Stream s1 synchronization timed out")
        
        print("\nEvent Operations:")
        print("----------------")
        
        # Create events
        e1 = torch.cuda.Event(enable_timing=True)
        print("Event created with timing enabled")
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Record the start event
        start_event.record()
        
        # Do some computation
        z = torch.matmul(x, y)
        
        # Record the end event
        end_event.record()
        
        # Synchronize to get accurate timing with timeout
        def sync_end_event():
            end_event.synchronize()
            return True
        
        if run_with_timeout(sync_end_event):
            # Calculate elapsed time
            try:
                elapsed_time = start_event.elapsed_time(end_event)
                print(f"Matrix multiplication took {elapsed_time:.2f} ms")
            except Exception as e:
                print(f"Error calculating elapsed time: {e}")
        else:
            print("Event synchronization timed out")
        
        print("\nStream Synchronization:")
        print("----------------------")
        
        # Create two streams
        s1 = torch.cuda.stream()
        s2 = torch.cuda.stream()
        
        # Record an event in s1
        e1 = s1.record_event()
        print("Event recorded in stream s1")
        
        # Make s2 wait for e1
        try:
            s2.wait_event(e1)
            print("Stream s2 waiting for event from Stream s1")
        except Exception as e:
            print(f"Error in wait_event: {e}")
        
        # Make one stream wait for another
        try:
            s2.wait_stream(s1)
            print("Stream s2 waiting for Stream s1")
        except Exception as e:
            print(f"Error in wait_stream: {e}")
        
        print("\nGeneral Stream API:")
        print("-----------------")
        
        # Using the general torch.Stream API
        if hasattr(torch, 'Stream'):
            try:
                s = torch.Stream(device=device)
                print(f"Created a general Stream for device: {device}")
                
                # Record an event
                e = s.record_event()
                print("Event recorded in the stream")
                
                # Synchronize the stream with timeout
                def sync_stream():
                    s.synchronize()
                    return "Stream synchronized"
                
                print(run_with_timeout(sync_stream) or "Stream synchronization timed out")
            except Exception as e:
                print(f"Error using torch.Stream: {e}")
        else:
            print("torch.Stream is not available in this PyTorch version")
        
    except Exception as e:
        print(f"Error in demo: {e}")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
